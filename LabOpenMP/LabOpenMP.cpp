#include <iostream>
#include <fstream>
#include <omp.h>
#include <algorithm>

#include <chrono>

#include "Matrix.h"

const int MATRIX_SEED = 123123;

int askMatrixSize() {
	int size = -1;

	std::cout << "Matrix size: ";

	while (size <= 0) {
		std::cin >> size;
		if (size <= 0) {
			std::cout << std::endl << "Enter positive integer" << std::endl;
		}
	}

	return size;
}

template<typename T>
void initExtMatrix(Matrix<T>* ext) {
	for (int i = 0; i < ext->size; i++) {
		for (int j = 0; j < ext->size; j++) {
			ext->main[i * ext->size + j] = (i == j) ? 1 : 0;
		}
	}
}

template<typename T>
int findMaxI(int step, Matrix<T>* mat) {
	T maxVal;
	int maxRow = -1;
	for (int i = step; i < mat->size; i++) {
		T val = mat->main[i * mat->size + step];
		if (maxRow == -1 || maxVal < std::abs(val)) {
			maxVal = std::abs(val);
			maxRow = i;
		}
	}

	return maxRow;
}

template<typename T>
void swapRows(int i1, int i2, Matrix<T>* mat) {
	for (int j = 0; j < mat->size; j++) {
		std::swap(mat->main[i1 * mat->size + j], mat->main[i2 * mat->size + j]);
	}
}

template<typename T>
void putMaxRowOnDiag(int step, Matrix<T>* mat, Matrix<T>* ext) {
	int maxI = findMaxI(step, mat);

	swapRows(step, maxI, mat);
	swapRows(step, maxI, ext);
}

template<typename T>
void normalizeLeadingCoef(int step, Matrix<T>* mat, Matrix<T>* ext) {
	T coefInv = 1 / mat->main[step * mat->size + step];

	for (int j = step; j < mat->size; j++) {
		mat->main[step * mat->size + j] *= coefInv;
	}

	mat->main[step * mat->size + step] = 1;

	for (int j = 0; j < mat->size; j++) {
		ext->main[step * mat->size + j] *= coefInv;
	}
}

template<typename T>
void firstGausInnerStep(int step, Matrix<T>* mat, T* maxMatRow, Matrix<T>* ext, T* maxExtRow) {

#pragma omp parallel for

	for (int i = step + 1; i < mat->size; i++) {
		T leadingCoef = mat->main[i * mat->size + step];

		for (int j = step; j < mat->size; j++) {
			mat->main[i * mat->size + j] -= leadingCoef * maxMatRow[j];
		}

		mat->main[i * mat->size + step] = 0;

		for (int j = 0; j < ext->size; j++) {
			ext->main[i * ext->size + j] -= leadingCoef * maxExtRow[j];
		}
	}
}

template<typename T>
bool tryMakeFirstStep(Matrix<T>* mat, Matrix<T>* ext) {
	T* maxMatRow = new T[mat->size];
	T* maxExtRow = new T[ext->size];

	for (int step = 0; step < mat->size; step++) {
		int rowOffset = step * mat->size;
		putMaxRowOnDiag(step, mat, ext);

		if (isZero(mat->main[rowOffset + step])) {
			delete[] maxMatRow, maxExtRow;
			return false;
		}

		normalizeLeadingCoef(step, mat, ext);

		std::copy(mat->main + rowOffset, mat->main + rowOffset + mat->size, maxMatRow);
		std::copy(ext->main + rowOffset, ext->main + rowOffset + ext->size, maxExtRow);

		firstGausInnerStep(step, mat, maxMatRow, ext, maxExtRow);
	}

	delete[] maxMatRow, maxExtRow;
	return true;
}

template<typename T>
void secondGausInnerStep(int step, Matrix<T>* mat, Matrix<T>* ext, T* extRow) {

#pragma omp parallel for

	for (int i = 0; i < step; i++) {
		T coef = mat->main[i * mat->size + step];
		mat->main[i * mat->size + step] = 0;

		for (int j = 0; j < ext->size; j++) {
			ext->main[i * ext->size + j] -= coef * extRow[j];
		}
	}
}

template<typename T>
void makeSecondGausStep(Matrix<T>* mat, Matrix<T>* ext) {
	T* extRow = new T[ext->size];
	for (int step = ext->size - 1; step >= 0; step--) {
		int rowOffset = step * mat->size;
		std::copy(ext->main + rowOffset, ext->main + rowOffset + ext->size, extRow);

		secondGausInnerStep(step, mat, ext, extRow);
	}

	delete[] extRow;
}

template<typename T>
void measureExecTime(const char* outCsvName) {
	std::fstream file;

	Matrix<T>* mat = nullptr;
	Matrix<T>* inv = nullptr;

	file.open(outCsvName, std::ios::out);

	file << "matrixSize, inverceDuration(milliseconds)" << std::endl;

	for (int size = 0; size <= 2000; size += 10) {
		mat = new Matrix<T>(size);
		fillWithRandom(MATRIX_SEED, size, mat->main);

		inv = new Matrix<T>(size);
		initExtMatrix(inv);

		auto start = std::chrono::high_resolution_clock::now();

		if (tryMakeFirstStep(mat, inv)) {
			makeSecondGausStep(mat, inv);
		}
		else {
			delete mat, inv;
			continue;
		}

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

		std::cout << size << "  :  " << duration.count() << std::endl;
		file << size << ", " << duration.count() << std::endl;

		delete mat, inv;
	}

	file.close();
}

void standartPipeline() {
	int size = askMatrixSize();
	Matrix<float>* mat = new Matrix<float>(size);
	fillWithRandom(MATRIX_SEED, size, mat->main);

	Matrix<float>* ext = new Matrix<float>(size);
	initExtMatrix(ext);

	if (tryMakeFirstStep(mat, ext)) {
		makeSecondGausStep(mat, ext);
		std::cout << *ext << std::endl;
	}
	else {
		std::cout << "Can't inverse matrix" << std::endl;
	}

	delete mat, ext;
}

int main() {

	measureExecTime<float>("sequential.csv");

	return 0;
}