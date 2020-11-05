//
// Created by liulei on 2020/5/21.
//

#ifndef RTF_FRAME_CONVERTERS_H
#define RTF_FRAME_CONVERTERS_H

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


#endif //RTF_FRAME_CONVERTERS_H
