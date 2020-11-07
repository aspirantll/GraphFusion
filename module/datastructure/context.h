//
// Created by liulei on 2020/6/6.
//

#ifndef GraphFusion_CONTEXT_H
#define GraphFusion_CONTEXT_H
#include <memory>
#include <vector>
#include "frame_types.h"

using namespace std;

namespace rtf {
    template <class T> class Context {

    protected:
        vector<shared_ptr<T>> frames;
    public:
        Context() {

        }

        Context(vector<shared_ptr<T>> frames) {
            this->frames = frames;
        }

        void addFrame(shared_ptr<T> frame) {
            frames.push_back(frame);
        }

        shared_ptr<T> getFrame(int index) {
            return frames[index];
        }

        int getFrameNum() {
            return frames.size();
        }
    };

    /** the context for registration */
    class ICPRegistrationContext: public Context<FrameRGBDT> {
    public:
        ICPRegistrationContext(){}
    };

    class BARegistrationContext: public Context<FrameRGBDT> {
    public:
        BARegistrationContext(){}
    };


}
#endif //GraphFusion_CONTEXT_H
