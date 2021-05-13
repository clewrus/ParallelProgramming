#include <iostream>
#include <mpi.h>
#include <algorithm>
#include <limits>
#include <chrono>
#include <fstream>

#include "Matrix.h"
#include "MatrixPiece.h"

int procCount = -1;
int procRank = -1;

const int ROW_SWAP_DIAGONAL_TO_MAX = 1;

const int MATRIX_SEED = 12345;

bool inline isMainProcess() {
	return procRank == 0;
}

int askMatrixSize() {
	int size = -1;

	if (isMainProcess()) {
		std::cout << "Matrix size: ";

		while (size <= 0) {
			std::cin >> size;
			if (size <= 0) {
				std::cout << std::endl << "Enter positive integer" << std::endl;
			}
		}
	}

	MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	return size;
}

template<typename T>
void initializeMainMatrix(Matrix<T>** mat, int size) {
	if (isMainProcess()) {
		*mat = new Matrix<T>(size);
		fillWithRandom(MATRIX_SEED, size, (*mat)->main);
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

template<typename T>
void fillPiece(Matrix<T>* mat, MatrixPiece<T>* piece) {
	int* sendDispl = nullptr;
	int* sendCount = nullptr;

	if (isMainProcess()) {
		sendDispl = new int[piece->pieceCount];
		sendCount = new int[piece->pieceCount];
	}

	int distribNeeded = (piece->originSize + piece->pieceCount - 1) / piece->pieceCount;
	for (int i = 0; i < distribNeeded; i++) {
		if (isMainProcess()) {
			for (int j = 0; j < piece->pieceCount; j++) {
				sendDispl[j] = piece->originSize * (i * piece->pieceCount + j);
				sendCount[j] = (i * piece->pieceCount + j < piece->originSize) ? piece->originSize : 0;
			}
		}

		T* sendBuff = (mat == nullptr) ? nullptr : mat->main;
		T* receiveBuf = piece->main + i * piece->originSize;
		int receiveCount = (i * piece->pieceCount + piece->pieceIndex < piece->originSize) ? piece->originSize : 0;

		MPI_Scatterv(sendBuff, sendCount, sendDispl, MPI_FLOAT, receiveBuf, receiveCount, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}

	if (isMainProcess()) {
		delete[] sendDispl;
		delete[] sendCount;
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

template<typename T>
void sampleMatrix(Matrix<T>* mat, Matrix<T>* extMat, MatrixPiece<T>* piece, bool sampleMainMatrix = false) {
	int* recvDispl = nullptr;
	int* recvCount = nullptr;

	if (isMainProcess()) {
		recvDispl = new int[piece->pieceCount];
		recvCount = new int[piece->pieceCount];
	}

	int receivesNeeded = (piece->originSize + piece->pieceCount - 1) / piece->pieceCount;
	for (int i = 0; i < receivesNeeded; i++) {
		if (isMainProcess()) {
			for (int j = 0; j < piece->pieceCount; j++) {
				int row = i * piece->pieceCount + j;
				recvDispl[j] = piece->originSize * row;
				recvCount[j] = (row < piece->originSize) ? piece->originSize : 0;
			}
		}

		bool hasRequestedRow = (i < piece->calcPieceHeight());
		int sendCount = (hasRequestedRow) ? piece->originSize : 0;
		int sendOffset = piece->originSize * i;

		T* sendBuff, * recvBuff;

		if (sampleMainMatrix) {
			sendBuff = (hasRequestedRow) ? piece->main + sendOffset : nullptr;
			recvBuff = (isMainProcess()) ? mat->main : nullptr;
			MPI_Gatherv(sendBuff, sendCount, MPI_FLOAT, recvBuff, recvCount, recvDispl, MPI_FLOAT, 0, MPI_COMM_WORLD);
		}

		sendBuff = (hasRequestedRow) ? piece->extended + sendOffset : nullptr;
		recvBuff = (isMainProcess()) ? extMat->main : nullptr;
		MPI_Gatherv(sendBuff, sendCount, MPI_FLOAT, recvBuff, recvCount, recvDispl, MPI_FLOAT, 0, MPI_COMM_WORLD);
	}

	if (isMainProcess()) {
		delete[] recvCount;
		delete[] recvDispl;
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

template<typename T>
void getPieceMaxRow(int step, MatrixPiece<T>* piece, int& maxRow, T& maxValue) {
	int pieceLeadingRow = (step + piece->pieceCount - piece->pieceIndex - 1) / piece->pieceCount;

	maxValue = std::numeric_limits<T>::min();
	int localMaxRow = -1;
	for (int i = pieceLeadingRow; i < piece->calcPieceHeight(); i++) {
		T value = piece->main[piece->originSize * i + step];
		if (localMaxRow < 0 || std::abs(value) > std::abs(maxValue)) {
			maxValue = value;
			localMaxRow = i;
		}
	}

	maxRow = (localMaxRow < 0) ? -1 : piece->getWorldPieceI(localMaxRow);
}

template<typename T>
void getWorldMaxRow(MatrixPiece<T>* piece, int localMaxRow, T localMaxVal, int& worldMaxRow) {
	int* piecesMaxRows = nullptr;
	T* piecesMaxValues = nullptr;

	if (isMainProcess()) {
		piecesMaxRows = new int[piece->pieceCount];
		piecesMaxValues = new T[piece->pieceCount];
	}

	MPI_Gather(&localMaxRow, 1, MPI_INT, piecesMaxRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Gather(&localMaxVal, 1, MPI_FLOAT, piecesMaxValues, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

	worldMaxRow = -1;
	T worldMaxElement = std::numeric_limits<T>::min();
	if (isMainProcess()) {
		for (int i = 0; i < piece->pieceCount; i++) {
			if (worldMaxRow < 0 || std::abs(worldMaxElement) < std::abs(piecesMaxValues[i])) {
				worldMaxElement = piecesMaxValues[i];
				worldMaxRow = piecesMaxRows[i];
			}
		}

		delete[] piecesMaxRows;
		delete[] piecesMaxValues;
	}

	MPI_Bcast(&worldMaxRow, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

template<typename T>
void swapRowsFromMaxPiece(MatrixPiece<T>* piece, int worldMaxRow, int destPiece) {
	T* diag = new T[2 * piece->originSize];
	MPI_Recv(diag, 2 * piece->originSize, MPI_FLOAT, destPiece, ROW_SWAP_DIAGONAL_TO_MAX, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);

	int offset = piece->getLocalPieceI(worldMaxRow) * piece->originSize;

	std::copy(diag, diag + piece->originSize, piece->main + offset);
	std::copy(diag + piece->originSize, diag + 2 * piece->originSize, piece->extended + offset);

	delete[] diag;
}

template<typename T>
void swapRowsFromDestPiece(MatrixPiece<T>* piece, T* maxRow, int step, int maxPiece) {
	int offset = piece->getLocalPieceI(step) * piece->originSize;

	T* diag = new T[2 * piece->originSize];
	std::copy(piece->main + offset, piece->main + offset + piece->originSize, diag);
	std::copy(piece->extended + offset, piece->extended + offset + piece->originSize, diag + piece->originSize);

	MPI_Send(diag, 2 * piece->originSize, MPI_FLOAT, maxPiece, ROW_SWAP_DIAGONAL_TO_MAX, MPI_COMM_WORLD);

	std::copy(maxRow, maxRow + piece->originSize, piece->main + offset);
	std::copy(maxRow + piece->originSize, maxRow + 2 * piece->originSize, piece->extended + offset);

	delete[] diag;
}

template<typename T>
void selectLeadingRow(int step, MatrixPiece<T>* piece, T* maxRowWithExtend, bool& swapHappened) {
	int localMaxRow, worldMaxRow;
	T localMaxValue;

	getPieceMaxRow(step, piece, localMaxRow, localMaxValue);
	getWorldMaxRow(piece, localMaxRow, localMaxValue, worldMaxRow);
	swapHappened = (worldMaxRow != step);

	int maxPiece = worldMaxRow % piece->pieceCount;
	int destPiece = step % piece->pieceCount;
	int destLocalRow = step / piece->pieceCount;

	if (piece->pieceIndex == maxPiece) {
		int offset = piece->getLocalPieceI(worldMaxRow) * piece->originSize;
		std::copy(piece->main + offset, piece->main + offset + piece->originSize, maxRowWithExtend);
		std::copy(piece->extended + offset, piece->extended + offset + piece->originSize, maxRowWithExtend + piece->originSize);
	}

	MPI_Bcast(maxRowWithExtend, 2 * piece->originSize, MPI_FLOAT, maxPiece, MPI_COMM_WORLD);

	if (destPiece == maxPiece) {
		if (piece->pieceIndex == destPiece) {
			piece->swapRows(step, worldMaxRow);
		}

		MPI_Barrier(MPI_COMM_WORLD);
		return;
	}

	if (piece->pieceIndex == maxPiece) {
		swapRowsFromMaxPiece(piece, worldMaxRow, destPiece);
	}

	if (piece->pieceIndex == destPiece) {
		swapRowsFromDestPiece(piece, maxRowWithExtend, step, maxPiece);
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

template<typename T>
void makeIteration(int step, MatrixPiece<T>* piece, T* maxRow) {
	T leadElementInv = 1 / maxRow[step];
	for (int i = step; i < 2 * piece->originSize; i++) {
		maxRow[i] *= leadElementInv;
	}
	maxRow[step] = 1;

	int pieceLeadingRow = (step + piece->pieceCount - piece->pieceIndex - 1) / piece->pieceCount;
	for (int i = pieceLeadingRow; i < piece->calcPieceHeight(); i++) {
		int offset = i * piece->originSize;

		if (piece->getWorldPieceI(i) == step) {
			std::copy(maxRow, maxRow + piece->originSize, piece->main + offset);
			std::copy(maxRow + piece->originSize, maxRow + 2 * piece->originSize, piece->extended + offset);
			continue;
		}

		T rowLead = (piece->main + offset)[step];
		for (int j = step; j < piece->originSize; j++) {
			(piece->main + offset)[j] -= rowLead * maxRow[j];
		}

		for (int j = 0; j < piece->originSize; j++) {
			(piece->extended + offset)[j] -= rowLead * (maxRow + piece->originSize)[j];
		}

		(piece->main + offset)[step] = 0;
	}
}

template<typename T>
bool tryMakeFirstGauseStep(MatrixPiece<T>* piece, T& determinant) {
	determinant = 1;
	bool swapHappened;

	T* maxRowWitExtent = new T[2 * piece->originSize];

	for (int i = 0; i < piece->originSize; i++) {
		selectLeadingRow(i, piece, maxRowWitExtent, swapHappened);

		determinant *= maxRowWitExtent[i];
		determinant *= (swapHappened) ? -1 : 1;

		if (isZero(maxRowWitExtent[i])) {
			delete[] maxRowWitExtent;
			return false;
		}

		makeIteration(i, piece, maxRowWitExtent);

		MPI_Barrier(MPI_COMM_WORLD);
	}

	delete[] maxRowWitExtent;

	MPI_Barrier(MPI_COMM_WORLD);
	return true;
}

template<typename T>
void getSecondStepExtendedRow(int step, MatrixPiece<T>* piece, T* extendedRow) {
	int rowHolderPiece = step % piece->pieceCount;

	if (piece->pieceIndex == rowHolderPiece) {
		int offset = piece->getLocalPieceI(step) * piece->originSize;
		T* src = piece->extended + offset;
		std::copy(src, src + piece->originSize, extendedRow);
	}

	MPI_Bcast(extendedRow, piece->originSize, MPI_FLOAT, rowHolderPiece, MPI_COMM_WORLD);
}

template<typename T>
void makeSecondGauseStep(MatrixPiece<T>* piece) {
	T* extendedRow = new T[2 * piece->originSize];

	for (int i = piece->originSize - 1; i >= 0; i--) {
		getSecondStepExtendedRow(i, piece, extendedRow);

		for (int j = piece->pieceIndex; j < i; j += piece->pieceCount) {
			int offset = piece->getLocalPieceI(j) * piece->originSize;

			T lastRowElem = (piece->main + offset)[i];
			(piece->main + offset)[i] = 0;

			for (int k = 0; k < piece->originSize; k++) {
				(piece->extended + offset)[k] -= extendedRow[k] * lastRowElem;
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}

	delete[] extendedRow;
}

template<typename T>
bool tryEvaluateInverce(int size, Matrix<T>* mat, Matrix<T>* inverce, T& det) {
	MatrixPiece<float>* piece = new MatrixPiece<float>(size, procRank, procCount);
	piece->initializeExtended();
	fillPiece(mat, piece);

	if (!tryMakeFirstGauseStep(piece, det)) {
		delete piece;
		return false;
	}

	makeSecondGauseStep(piece);
	sampleMatrix(mat, inverce, piece);

	delete piece;
	return true;
}

template<typename T>
void measureExecTime(const char* outCsvName) {
	std::fstream file;

	T det;
	Matrix<T>* mat = nullptr;
	Matrix<T>* inv = nullptr;

	if (isMainProcess()) {
		file.open(outCsvName, std::ios::out);
		file << "matrixSize, inverceDuration(milliseconds)" << std::endl;
		file << 0 << ", " << 0 << std::endl;
	}

	for (int size = 10; size <= 2000; size += 10) {
		MPI_Barrier(MPI_COMM_WORLD);
		auto start = std::chrono::high_resolution_clock::now();

		initializeMainMatrix(&mat, size);

		if (isMainProcess()) {
			inv = new Matrix<float>(size);
		}

		if (!tryEvaluateInverce(size, mat, inv, det)) {
			if (isMainProcess()) {
				std::cout << "Matrix has no inverce" << std::endl;
			}
			MPI_Barrier(MPI_COMM_WORLD);
			continue;
		}

		MPI_Barrier(MPI_COMM_WORLD);
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		if (isMainProcess()) {
			std::cout << size << "  :  " << duration.count() << std::endl;
			file << size << ", " << duration.count() << std::endl;
		}
		
		delete mat, inv;
	}

	if (isMainProcess()) {
		file.close();
	}
}

int main(int argc, char** argv) {

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &procCount);
	MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

	measureExecTime<float>("mpiExecTest.csv");

	MPI_Finalize();
	return 0;

	Matrix<float>* mat = nullptr;
	Matrix<float>* inverted = nullptr;
	float det;

//	int size = askMatrixSize();

	for (int size = 2000; size < 2001; size++) {
		initializeMainMatrix(&mat, size);

		if (isMainProcess()) {
			inverted = new Matrix<float>(size);
		}

		if (!tryEvaluateInverce(size, mat, inverted, det)) {
			if (isMainProcess()) {
				std::cout << "Matrix has no inverce" << std::endl;
			}
			continue;
			return 0;
		}

		MPI_Barrier(MPI_COMM_WORLD);

		if (isMainProcess()) {
			std::cout << "testing" << std::endl;
			std::cout << "Is valid (" << size << "): " << testInverce(mat, inverted) << std::endl;
			delete mat;
			delete inverted;
		}

		MPI_Barrier(MPI_COMM_WORLD);
	}
	

	MPI_Finalize();
	return 0;
}
