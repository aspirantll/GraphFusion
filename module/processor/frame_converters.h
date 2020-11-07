//
// Created by liulei on 2020/5/21.
//

#ifndef GraphFusion_FRAME_CONVERTERS_H
#define GraphFusion_FRAME_CONVERTERS_H

#include "../datastructure/frame_types.h"

namespace rtf {
    namespace FrameConverters {

        /**
         * convert depth to float, rgb to rgba
         * @param frame
         */
        void convertImageType(shared_ptr<FrameRGBD> frame);

    }
}


#endif //GraphFusion_FRAME_CONVERTERS_H
