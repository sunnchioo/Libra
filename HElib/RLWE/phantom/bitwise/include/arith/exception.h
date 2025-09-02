#pragma once

#include <stdexcept>
#include <string>

namespace isecfhe {
    class FHEException : public std::runtime_error {
        std::string fileName;
        int lineNum;
        std::string message;

    public:
        FHEException(const std::string &file, int line, const std::string &what)
                : std::runtime_error(what), fileName(file), lineNum(line) {
            message = fileName + ":" + std::to_string(lineNum) + " " + what;
        }

        const char *what() const throw() { return message.c_str(); }

        const std::string &GetFileName() const { return fileName; }

        int GetLineNum() const { return lineNum; }
    };

    class ParamException : public FHEException {
    public:
        ParamException(const std::string &file, int line, const std::string &what)
                : FHEException(file, line, what) {}
    };


    class ConfigException : public FHEException {
    public:
        ConfigException(const std::string &file, int line, const std::string &what)
                : FHEException(file, line, what) {}
    };

    class MathException : public FHEException {
    public:
        MathException(const std::string &file, int line, const std::string &what)
                : FHEException(file, line, what) {}
    };

    class NotImplementedException : public FHEException {
    public:
        NotImplementedException(const std::string &file, int line, const std::string &what)
                : FHEException(file, line, what) {}
    };

    class NotAvailableError : public FHEException {
    public:
        NotAvailableError(const std::string &file, int line, const std::string &what)
                : FHEException(file, line, what) {}
    };

    class TypeException : public FHEException {
    public:
        TypeException(const std::string &file, int line, const std::string &what)
                : FHEException(file, line, what) {}
    };


    class SerializeError : public FHEException {
    public:
        SerializeError(const std::string &file, int line, const std::string &what)
                : FHEException(file, line, what) {}
    };


    class DeserializeError : public FHEException {
    public:
        DeserializeError(const std::string &file, int line, const std::string &what)
                : FHEException(file, line, what) {}
    };

#define FHE_THROW(exc, expr) throw exc(__FILE__, __LINE__, (expr))
}// namespace isecfhe
