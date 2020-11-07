//
// Created by liulei on 2020/7/12.
//

#ifndef GraphFusion_TIMER_H
#define GraphFusion_TIMER_H


#include <ctime>
#include <string>
#include <utility>
#include <iostream>

using namespace std;

namespace rtf {
    class Timer {
    private:
        string name;
        time_t start;
    public:
        static Timer startTimer(string name) {
            Timer timer;
            timer.name = std::move(name);
            timer.start = clock();
            return timer;
        }

        void stopTimer() {
            cout << name << "'s time cost:" << double(clock()-start)/CLOCKS_PER_SEC << endl;
        }
    };
}


#endif //GraphFusion_TIMER_H
