#pragma once
#include <chrono>
#include <iostream>

class Timer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_;

    template <typename T = std::chrono::milliseconds>
    inline void print_duration(const char *text) {
        std::cout << text << std::chrono::duration_cast<T>(end_ - start_).count() << std::endl;
    }

public:
    Timer() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    inline void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    template <typename T = std::chrono::milliseconds>
    inline void stop(const char *text) {
        end_ = std::chrono::high_resolution_clock::now();
        print_duration<T>(text);
    }

    inline void stop() {
        end_ = std::chrono::high_resolution_clock::now();
    }

    template <typename T = std::chrono::milliseconds>
    inline long timer_duration() {
        return std::chrono::duration_cast<T>(end_ - start_).count();
    }
};
