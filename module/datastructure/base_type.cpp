//
// Created by liulei on 2020/9/20.
//

#include "base_types.h"

namespace rtf {

    Runnable::Runnable(): mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mbStopped(false)
                , mbStopRequested(false), mbNotStop(false), mbAcceptRequest(true){

    }

    void Runnable::resetIfRequested() {
        unique_lock<mutex> lock(mMutexReset);
        if (mbResetRequested) {
            reset();
            mbResetRequested = false;
        }
    }

    void Runnable::requestFinish() {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    bool Runnable::checkFinish() {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void Runnable::setFinish() {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = true;
        unique_lock<mutex> lock2(mMutexStop);
        mbStopped = true;
    }

    bool Runnable::isFinished() {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinished;
    }

    void Runnable::requestStop() {
        unique_lock<mutex> lock(mMutexStop);
        mbStopRequested = true;
    }

    bool Runnable::stop() {
        unique_lock<mutex> lock(mMutexStop);
        if (mbStopRequested && !mbNotStop) {
            mbStopped = true;
            return true;
        }

        return false;
    }

    bool Runnable::isStopped() {
        unique_lock<mutex> lock(mMutexStop);
        return mbStopped;
    }

    void Runnable::requestReset() {
        {
            unique_lock<mutex> lock(mMutexReset);
            mbResetRequested = true;
        }

        while (1) {
            {
                unique_lock<mutex> lock2(mMutexReset);
                if (!mbResetRequested)
                    break;
            }
            usleep(3000);
        }
    }

    bool Runnable::stopRequested() {
        unique_lock<mutex> lock(mMutexStop);
        return mbStopRequested;
    }

    bool Runnable::setNotStop(bool flag) {
        unique_lock<mutex> lock(mMutexStop);

        if (flag && mbStopped)
            return false;

        mbNotStop = flag;

        return true;
    }

    bool Runnable::acceptRequest()
    {
        unique_lock<mutex> lock(mMutexAccept);
        return mbAcceptRequest;
    }

    void Runnable::setAcceptRequest(bool flag)
    {
        unique_lock<mutex> lock(mMutexAccept);
        mbAcceptRequest=flag;
    }

    void Runnable::run() {
        mbFinished = false;

        while(1)
        {
            // Tracking will see that Local Mapping is busy
            setAcceptRequest(false);

            // Check if there are viewclusters in the queue
            if(checkNewRequest())
            {
                doTask();
            }
            else if(stop())
            {
                // Safe area to stop
                while(isStopped() && !checkFinish())
                {
                    usleep(3000);
                }
                if(checkFinish())
                    break;
            }

            resetIfRequested();

            // Tracking will see that Local Mapping is busy
            setAcceptRequest(true);

            if(checkFinish())
                break;

            usleep(3000);
        }

        setFinish();
    }

}
