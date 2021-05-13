#pragma once

#include <iostream>
#include <iomanip>

template<typename T>
struct MatrixPiece {
	int originSize;

	int pieceIndex;
	int pieceCount;

	T* main = nullptr;
	T* extended = nullptr;

	MatrixPiece(int originSize, int pieceIndex, int pieceCount) :
		originSize(originSize), pieceIndex(pieceIndex), pieceCount(pieceCount)
	{
		main = new T[originSize * calcPieceHeight()];
		extended = new T[originSize * calcPieceHeight()];
	}

	~MatrixPiece() {
		delete[] main;
		delete[] extended;
	}

	int inline calcPieceHeight() {
		return originSize / pieceCount + (pieceIndex < originSize % pieceCount);
	}

	int inline getLocalPieceI(int originI) const {
		return originI / pieceCount;
	}

	int inline getWorldPieceI(int localI) const {
		return localI * pieceCount + pieceIndex;
	}

	void initializeExtended() {
		for (int i = pieceIndex; i < originSize; i += pieceCount) {
			for (int j = 0; j < originSize; j++) {
				extended[getLocalPieceI(i) * originSize + j] = (i == j) ? 1 : 0;
			}
		}
	}

	void swapRows(int worldA, int worldB) {
		if (worldA == worldB) { return; }
		int offsetA = originSize * getLocalPieceI(worldA);
		int offsetB = originSize * getLocalPieceI(worldB);

		for (int i = 0; i < originSize; i++) {
			std::swap(main[offsetA + i], main[offsetB + i]);
			std::swap(extended[offsetA + i], extended[offsetB + i]);
		}
	}
};

template<typename T>
std::ostream& operator<<(std::ostream& stream, const MatrixPiece<T>& piece) {
	stream << std::scientific << std::setprecision(3);

	for (int i = piece.pieceIndex; i < piece.originSize; i += piece.pieceCount) {
		stream << std::setw(5) << i << ": ";
		for (int j = 0; j < piece.originSize; j++) {
			stream << std::setw(12) << piece.main[piece.getLocalPieceI(i) * piece.originSize + j] << " ";
		}

		stream << "| ";

		for (int j = 0; j < piece.originSize; j++) {
			stream << std::setw(12) << piece.extended[piece.getLocalPieceI(i) * piece.originSize + j] << " ";
		}

		stream << std::endl;
	}

	return stream;
}