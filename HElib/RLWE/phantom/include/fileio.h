#pragma once

#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

template <typename T>
class FileIO {
public:
    // ======================= 二进制格式读写 =======================
    static void SaveBinary(const std::string &filename, const std::vector<T> &data) {
        std::ofstream file(filename, std::ios::binary);
        CheckFileOpen(file, filename);

        // 写入数据长度 (兼容32/64位)
        uint64_t size = data.size();
        file.write(reinterpret_cast<const char *>(&size), sizeof(size));

        // 写入数据内容
        if (!data.empty()) {
            file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(T));
        }

        CheckWriteOperation(file, filename);
    }

    static std::vector<T> LoadBinary(const std::string &filename) {
        std::ifstream file(filename, std::ios::binary);
        CheckFileOpen(file, filename);

        // 读取数据长度
        uint64_t size;
        file.read(reinterpret_cast<char *>(&size), sizeof(size));
        CheckReadOperation(file, filename, "size header");

        // 预分配内存
        std::vector<T> data(size);

        // 读取数据内容
        if (size > 0) {
            file.read(reinterpret_cast<char *>(data.data()), size * sizeof(T));
            CheckReadOperation(file, filename, "data content");
        }

        return data;
    }

    // ======================= 文本格式读写 =======================
    // 浮点类型特化：支持精度参数
    template <typename U = T>
    static std::enable_if_t<std::is_floating_point_v<U>>
    SaveText(const std::string &filename,
             const std::vector<U> &data,
             int precision = std::numeric_limits<U>::max_digits10) {
        std::ofstream file(filename);
        CheckFileOpen(file, filename);
        file << std::setprecision(precision);
        WriteTextData(file, data);
    }

    // 整数类型特化：无精度参数
    template <typename U = T>
    static std::enable_if_t<!std::is_floating_point_v<U>>
    SaveText(const std::string &filename, const std::vector<U> &data) {
        std::ofstream file(filename);
        CheckFileOpen(file, filename);
        WriteTextData(file, data);
    }

    static std::vector<T> LoadText(const std::string &filename) {
        std::ifstream file(filename);
        CheckFileOpen(file, filename);

        std::vector<T> data;
        T val;
        while (file >> val) {
            data.push_back(val);
        }

        if (!file.eof()) {
            throw std::runtime_error("Failed to parse text file: " + filename);
        }

        return data;
    }

    // ================== 新增三维CSV文件读取 ==================
    template <typename U = T>
    static std::enable_if_t<std::is_same_v<U, double>,
                            std::vector<std::vector<U>>>
    LoadCSV3D(const std::string &filename) {
        std::ifstream file(filename);
        CheckFileOpen(file, filename);
        // std::vector<std::vector<double>> data;
        std::vector<std::vector<U>> data;
        std::string line;
        int line_num = 0;
        while (std::getline(file, line)) {
            ++line_num;
            std::istringstream line_stream(line);
            std::vector<U> point(3, 0);
            char sep;
            // 带格式验证的流解析
            if (!(line_stream >> point[0] >> sep >> point[1] >> sep >> point[2])) {
                throw ParseError(line_num, "Invalid format (insufficient elements or conversion failure)");
            }
            // 统一验证分隔符
            if (sep != ',') {
                throw ParseError(line_num, "Separator must be comma");
            }
            // 检查行尾冗余内容
            std::string remaining;
            if (line_stream >> remaining) {
                throw ParseError(line_num, "Detected trailing content:" + remaining);
            }
            std::vector<U> new_row(point.end() - 2, point.end());
            data.push_back(new_row);
        }
        if (file.bad()) {
            throw std::runtime_error("File read error: " + filename);
        }
        return data;
    }

private:
    // ======================= 公共工具函数 =======================
    static void WriteTextData(std::ofstream &file, const std::vector<T> &data) {
        for (const auto &val : data) {
            file << val << "\n";
        }
        CheckWriteOperation(file, file.is_open() ? "open stream" : "closed stream");
    }

    // ======================= 错误处理 =======================
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

    static void CheckWriteOperation(const std::ofstream &file, const std::string &context) {
        if (!file.good()) {
            throw std::runtime_error("Write operation failed: " + context);
        }
    }

    static void CheckReadOperation(const std::ifstream &file,
                                   const std::string &filename,
                                   const std::string &context) {
        if (!file.good()) {
            throw std::runtime_error("Read operation failed at " + context + ": " + filename);
        }
    }

    // ================== 新增异常处理类 ==================
    class ParseError : public std::runtime_error {
    public:
        ParseError(int line, const std::string &msg)
            : runtime_error("[Line " + std::to_string(line) + "] " + msg) {}
    };
};
