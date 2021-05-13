#include "Matrix.h"

template<typename T>
Matrix<T>::Matrix(int size) : size(size) {
	main = new T[size * size];
}

template<typename T>
Matrix<T>::~Matrix() {
	delete[] main;
}

template<typename T>
std::ostream& operator<<(std::ostream& stream, const Matrix<T>& mat) {
	stream << std::scientific << std::setprecision(3);

	for (int i = 0; i < mat.size; i++) {
		stream << std::setw(5) << i << ": ";
		for (int j = 0; j < mat.size; j++) {
			stream << std::setw(12) << mat.main[i * mat.size + j] << " ";
		}
		stream << std::endl;
	}

	return stream;
}

template<typename T>
void fillWithRandom(int seed, int size, T* matrix) {
	srand(seed);
	for (int i = 0; i < size * size; i++) {
		T r = static_cast<T>(rand()) / RAND_MAX;
		matrix[i] = 200 * r - 100;
	}
}