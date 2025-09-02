#pragma once

// #include <Eigen/Sparse>
#include <complex>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

// using Complex = std::complex<double>;
// using SparseMatrix = Eigen::SparseMatrix<Complex>;
// using Triplet = Eigen::Triplet<Complex>;
// class SparseMatrixHandler {
// public:
//     // 构造函数：可以空构造或直接用矩阵构造
//     SparseMatrixHandler() = default;
//     SparseMatrixHandler(const SparseMatrix &mat) : matrix_(mat) {}

//     // 设置矩阵
//     void setMatrix(const SparseMatrix &mat) {
//         matrix_ = mat;
//     }

//     // 获取矩阵
//     const SparseMatrix &getMatrix() const {
//         return matrix_;
//     }

//     // 保存稀疏矩阵到文本文件
//     bool saveToText(const std::string &filename) const {
//         if (matrix_.nonZeros() == 0) {
//             std::cerr << "Warning: Saving empty matrix." << std::endl;
//         }

//         std::ofstream outfile(filename);
//         if (!outfile.is_open()) {
//             std::cerr << "Error opening file: " << filename << std::endl;
//             return false;
//         }

//         outfile.precision(15); // 高精度输出

//         // 写入矩阵维度信息
//         outfile << matrix_.rows() << " "
//                 << matrix_.cols() << " "
//                 << matrix_.nonZeros() << "\n";

//         // 遍历所有非零元素
//         for (int k = 0; k < matrix_.outerSize(); ++k) {
//             for (typename SparseMatrix::InnerIterator it(matrix_, k); it; ++it) {
//                 const Complex &value = it.value();
//                 outfile << it.row() << " "
//                         << it.col() << " "
//                         << value.real() << " "
//                         << value.imag() << "\n";
//             }
//         }

//         return true;
//     }

//     // 从文本文件加载稀疏矩阵
//     bool loadFromText(const std::string &filename) {
//         std::ifstream infile(filename);
//         if (!infile.is_open()) {
//             std::cerr << "Error opening file: " << filename << std::endl;
//             return false;
//         }

//         int rows, cols, nnz;
//         infile >> rows >> cols >> nnz;

//         std::vector<Triplet> triplets;
//         triplets.reserve(nnz);

//         for (int i = 0; i < nnz; ++i) {
//             int row, col;
//             double real, imag;
//             infile >> row >> col >> real >> imag;
//             triplets.emplace_back(row, col, Complex(real, imag));
//         }

//         matrix_ = SparseMatrix(rows, cols);
//         matrix_.setFromTriplets(triplets.begin(), triplets.end());
//         matrix_.makeCompressed();

//         return true;
//     }

// private:
//     SparseMatrix matrix_;
// };

class File {
public:
    // 可选配置：设置分隔符和大小写
    void setSeparator(const std::string &sep) { separator_ = sep; }
    void setUpperCase(bool enable) { uppercase_ = enable; }

    // 核心写入方法
    void write(const std::string &filename, const uint64_t *data, size_t count) {
        std::ofstream file(filename, std::ios::binary);
        if (!file)
            throw std::runtime_error("无法打开文件: " + filename);

        std::ostringstream buffer;
        buffer << std::hex << (uppercase_ ? std::uppercase : std::nouppercase);

        for (size_t i = 0; i < count; ++i) {
            buffer.str(""); // 清空缓冲区
            buffer << std::setw(16) << std::setfill('0') << data[i];
            file << buffer.str() << separator_;
        }
    }

    // 向量版本重载
    void write(const std::string &filename, const std::vector<uint64_t> &data) {
        write(filename, data.data(), data.size());
    }

private:
    std::string separator_ = ""; // 默认无分隔符
    bool uppercase_ = true;      // 默认大写字母
};

class FileIO {
public:
    // ======================= 二进制格式读写 =======================
    // 高性能，适合大规模数据
    static void SaveBinary(const std::string &filename, const std::vector<double> &data) {
        std::ofstream file(filename, std::ios::binary);
        CheckFileOpen(file, filename);

        // 写入数据长度 (兼容32/64位)
        uint64_t size = data.size();
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));

        // 写入数据内容
        if (!data.empty()) {
            file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(double));
        }

        CheckWriteOperation(file, filename);
    }

    static std::vector<double> LoadBinary(const std::string &filename) {
        std::ifstream file(filename, std::ios::binary);
        CheckFileOpen(file, filename);

        // 读取数据长度
        uint64_t size;
        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        CheckReadOperation(file, filename, "size header");

        // 预分配内存
        std::vector<double> data(size);

        // 读取数据内容
        if (size > 0) {
            file.read(reinterpret_cast<char *>(data.data()), size * sizeof(double));
            CheckReadOperation(file, filename, "data content");
        }

        return data;
    }

    // ======================= 文本格式读写 =======================
    static void SaveText(const std::string &filename,
                         const std::vector<double> &data,
                         int precision = std::numeric_limits<double>::max_digits10) {
        std::ofstream file(filename);
        CheckFileOpen(file, filename);

        // 设置最高精度
        file << std::setprecision(precision);

        for (const auto &val : data) {
            file << val << "\n";
        }

        CheckWriteOperation(file, filename);
    }

    static std::vector<double> LoadText(const std::string &filename) {
        std::ifstream file(filename);
        CheckFileOpen(file, filename);

        std::vector<double> data;
        double val;

        while (file >> val) {
            data.push_back(val);
        }

        // 检查是否正常读完
        if (!file.eof()) {
            throw std::runtime_error("Failed to parse text file: " + filename);
        }

        return data;
    }

    static void saveComplexVector2D(const vector<vector<complex<double>>> &data,
                                    const string &filename) {
        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            cerr << "Error opening file: " << filename << endl;
            return;
        }

        // 设置高精度输出（15位小数）
        outfile << std::fixed << setprecision(15);

        // 写入标题行（可选）
        outfile << "row,col,real,imag" << endl;

        // 遍历所有元素
        for (int row = 0; row < data.size(); row++) {
            for (int col = 0; col < data[row].size(); col++) {
                const std::complex<double> &c = data[row][col];

                // 写入 row,col,real,imag
                outfile << row << ","
                        << col << ","
                        << c.real() << ","
                        << c.imag();

                // 每行一个条目
                outfile << "\n";
            }
        }

        outfile.close();
        std::cout << "Data saved to: " << filename << std::endl;
    }

private:
    // ======================= 错误处理工具函数 =======================
    static void CheckFileOpen(const std::ifstream &file, const std::string &filename) {
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for reading: " + filename);
        }
    }

    static void CheckFileOpen(const std::ofstream &file, const std::string &filename) {
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }
    }

    static void CheckWriteOperation(const std::ofstream &file, const std::string &filename) {
        if (!file.good()) {
            throw std::runtime_error("Write operation failed: " + filename);
        }
    }

    static void CheckReadOperation(const std::ifstream &file,
                                   const std::string &filename,
                                   const std::string &context) {
        if (!file.good()) {
            throw std::runtime_error("Read operation failed at " + context + ": " + filename);
        }
    }
};

class FileIOUint {
public:
    // ======================= 二进制格式读写 =======================
    static void SaveBinary(const std::string &filename, const std::vector<uint64_t> &data) {
        std::ofstream file(filename, std::ios::binary);
        CheckFileOpen(file, filename);

        // 写入数据长度 (兼容32/64位)
        uint64_t size = data.size();
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));

        // 写入数据内容
        if (!data.empty()) {
            file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(uint64_t));
        }

        CheckWriteOperation(file, filename);
    }

    static std::vector<uint64_t> LoadBinary(const std::string &filename) {
        std::ifstream file(filename, std::ios::binary);
        CheckFileOpen(file, filename);

        // 读取数据长度
        uint64_t size;
        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        CheckReadOperation(file, filename, "size header");

        // 预分配内存
        std::vector<uint64_t> data(size);

        // 读取数据内容
        if (size > 0) {
            file.read(reinterpret_cast<char *>(data.data()), size * sizeof(uint64_t));
            CheckReadOperation(file, filename, "data content");
        }

        return data;
    }

    // ======================= 文本格式读写 =======================
    static void SaveText(const std::string &filename,
                         const std::vector<uint64_t> &data) { // 移除浮点精度参数
        std::ofstream file(filename);
        CheckFileOpen(file, filename);

        // 直接写入整数，无需设置精度
        for (const auto &val : data) {
            file << val << "\n";
        }

        CheckWriteOperation(file, filename);
    }

    static std::vector<uint64_t> LoadText(const std::string &filename) {
        std::ifstream file(filename);
        CheckFileOpen(file, filename);

        std::vector<uint64_t> data;
        uint64_t val;

        while (file >> val) { // 直接读取 uint64_t 类型
            data.push_back(val);
        }

        // 检查是否正常读完
        if (!file.eof()) {
            throw std::runtime_error("Failed to parse text file: " + filename);
        }

        return data;
    }

private:
    // ======================= 错误处理工具函数 =======================
    static void CheckFileOpen(const std::ifstream &file, const std::string &filename) {
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for reading: " + filename);
        }
    }

    static void CheckFileOpen(const std::ofstream &file, const std::string &filename) {
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }
    }

    static void CheckWriteOperation(const std::ofstream &file, const std::string &filename) {
        if (!file.good()) {
            throw std::runtime_error("Write operation failed: " + filename);
        }
    }

    static void CheckReadOperation(const std::ifstream &file,
                                   const std::string &filename,
                                   const std::string &context) {
        if (!file.good()) {
            throw std::runtime_error("Read operation failed at " + context + ": " + filename);
        }
    }
};
