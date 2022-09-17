
#include "include/batch_headers/data_types.cl"

#define SORT_RESULT_CLASSID 0
#define SORT_RESULT_SCORE 1

#define INPUT_INDICES_TYPE int32

#ifndef HAS_ROISNUM

// KERNEL(whats_your_name_again)
//(const __global INPUT0_TYPE* boxes,
//  const __global INPUT0_TYPE* scores,
//  __global OUTPUT_INDICES_TYPE* selected_indices,
//  __global OUTPUT_INDICES_TYPE* selected_num,
//  __global OUTPUT_TYPE* selected_outputs) {
//
//     const  OUTPUT_TYPE selected_outputs_[] = {
//         0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 1.00, 0.95,
//         0.00, 0.00, 1.00, 1.00,  0.00, 0.90,  0.00, 0.00,
//         1.00, 1.00, 1.00, 0.80,  0.00, 10.00, 1.00, 11.00};
//
//     const  OUTPUT_INDICES_TYPE selected_indices_[] = {3, 0, 0, 3};
//
//     int n = 0;
//     for (; n < 4; ++n) {
//         for (int i = 0; i < 6; ++i) {
//             selected_outputs[6 * n + i] = selected_outputs_[6 * n + i];
//         }
//         selected_indices[n] = selected_indices_[n];
//     }
//     *selected_num = 4;
//     for (; n < OUTPUT_DIM; ++n) {
//         for (int i = 0; i < 6; ++i) {
//             selected_outputs[6 * n + i] = 0;
//         }
//         selected_indices[n] = 0;
//     }
//
//     //barrier(CLK_GLOBAL_MEM_FENCE);
//
//     printf("Two 2 inputs\n");
// }

typedef struct __attribute__((__packed__)) {
    INPUT0_TYPE score;
    INPUT0_TYPE xmin;
    INPUT0_TYPE ymin;
    INPUT0_TYPE xmax;
    INPUT0_TYPE ymax;
    OUTPUT_INDICES_TYPE class_idx;
    OUTPUT_INDICES_TYPE batch_idx;
    OUTPUT_INDICES_TYPE index;

} FUNC(BOX_INFO);

#    define BoxInfo FUNC(BOX_INFO)

inline void FUNC(swap_info)(__global BoxInfo* a, __global BoxInfo* b) {
    const BoxInfo temp = *a;
    *a = *b;
    *b = temp;
}

#define SORTMODE_CLASS 0
#define SORTMODE_SCORE_THEN_INDEX 1
#define SORTMODE_SCORE_THEN_CLASS 2

inline int FUNC(partition)(__global BoxInfo* arr, int l, int h, int sortMode) {
    const BoxInfo pivot = arr[h];

    int i = (l - 1);
    for (int j = l; j <= h - 1; j++) {
        switch(sortMode) {
            case SORTMODE_CLASS: {
                if ((arr[j].class_idx < pivot.class_idx) ||
                    (arr[j].class_idx == pivot.class_idx && arr[j].batch_idx < pivot.batch_idx) ||
                    (arr[j].class_idx == pivot.class_idx && arr[j].batch_idx == pivot.batch_idx &&
                     arr[j].score > pivot.score) ||
                    (arr[j].class_idx == pivot.class_idx && arr[j].batch_idx == pivot.batch_idx &&
                     arr[j].score == pivot.score && arr[j].index < pivot.index)) {
                    i++;
                    FUNC_CALL(swap_info)(&arr[i], &arr[j]);
                }
                break;
            }
            case SORTMODE_SCORE_THEN_INDEX: {
                if ((arr[j].score > pivot.score) || (arr[j].score == pivot.score && arr[j].index < pivot.index) ||
                    (arr[j].score == pivot.score && arr[j].index == pivot.index &&
                     arr[j].class_idx > pivot.class_idx) ||
                    (arr[j].score == pivot.score && arr[j].index == pivot.index &&
                     arr[j].class_idx == pivot.class_idx && arr[j].batch_idx > pivot.batch_idx)) {
                    i++;
                    FUNC_CALL(swap_info)(&arr[i], &arr[j]);
                }
                break;
            }
            case SORTMODE_SCORE_THEN_CLASS: {
                if ( (arr[j].batch_idx == pivot.batch_idx) &&
                     ((arr[j].score > pivot.score) || (arr[j].score == pivot.score && arr[j].class_idx < pivot.class_idx) ||
                     (arr[j].score == pivot.score && arr[j].class_idx == pivot.class_idx && arr[j].index < pivot.index))) {
                    i++;
                    FUNC_CALL(swap_info)(&arr[i], &arr[j]);
                }
                break;
            }
        } // switch
    }
    FUNC_CALL(swap_info)(&arr[i + 1], &arr[h]);
    return (i + 1);
}

inline void FUNC(bubbleSortIterative)(__global BoxInfo* arr, int l, int h, int sortMode) {
    for (int i = 0; i < h - l; i++) {
        bool swapped = false;
        for (int j = l; j < h - i; j++) {
            switch(sortMode) {
                case SORTMODE_CLASS: {
                    if ((arr[j].class_idx < arr[j + 1].class_idx) ||
                        (arr[j].class_idx == arr[j + 1].class_idx && arr[j].batch_idx < arr[j + 1].batch_idx) ||
                        (arr[j].class_idx == arr[j + 1].class_idx && arr[j].batch_idx == arr[j + 1].batch_idx &&
                         arr[j].score > arr[j + 1].score) ||
                        (arr[j].class_idx == arr[j + 1].class_idx && arr[j].batch_idx == arr[j + 1].batch_idx &&
                         arr[j].score == arr[j + 1].score && arr[j].index < arr[j + 1].index)) {
                        FUNC_CALL(swap_info)(&arr[j], &arr[j + 1]);
                        swapped = true;
                    }
                    break;
                }
                case SORTMODE_SCORE_THEN_INDEX: {
                    if ((arr[j].score > arr[j + 1].score) ||
                        (arr[j].score == arr[j + 1].score && arr[j].index < arr[j + 1].index) ||
                        (arr[j].score == arr[j + 1].score && arr[j].index == arr[j + 1].index &&
                         arr[j].class_idx < arr[j + 1].class_idx) ||
                        (arr[j].score == arr[j + 1].score && arr[j].index == arr[j + 1].index &&
                         arr[j].class_idx == arr[j + 1].class_idx && arr[j].batch_idx < arr[j + 1].batch_idx)) {
                        FUNC_CALL(swap_info)(&arr[j], &arr[j + 1]);
                        swapped = true;
                    }
                    break;
                }
                case SORTMODE_SCORE_THEN_CLASS: {
                    if ( (arr[j].batch_idx == arr[j + 1].batch_idx) &&
                         ((arr[j].score > arr[j + 1].score) || (arr[j].score == arr[j + 1].score && arr[j].class_idx < arr[j + 1].class_idx) ||
                         (arr[j].score == arr[j + 1].score && arr[j].class_idx == arr[j + 1].class_idx && arr[j].index < arr[j + 1].index))) {
                        FUNC_CALL(swap_info)(&arr[j], &arr[j + 1]);
                        swapped = true;
                    }
                    break;
                }
            } // switch
        }

        if (!swapped)
            break;
    }
}

inline void FUNC(quickSortIterative)(__global BoxInfo* arr, int l, int h, int sortMode) {
    // Create an auxiliary stack
    const int kStackSize = 100;
    int stack[kStackSize];

    // initialize top of stack
    int top = -1;

    // push initial values of l and h to stack
    stack[++top] = l;
    stack[++top] = h;

    // Keep popping from stack while is not empty
    while (top >= 0) {
        // Pop h and l
        h = stack[top--];
        l = stack[top--];

        // Set pivot element at its correct position
        // in sorted array
        int p = FUNC_CALL(partition)(arr, l, h, sortMode);

        // If there are elements on left side of pivot,
        // then push left side to stack
        if (p - 1 > l) {
            if (top >= (kStackSize - 1)) {
                FUNC_CALL(bubbleSortIterative)(arr, l, p - 1, sortMode);
            } else {
                stack[++top] = l;
                stack[++top] = p - 1;
            }
        }

        // If there are elements on right side of pivot,
        // then push right side to stack
        if (p + 1 < h) {
            if (top >= (kStackSize - 1)) {
                FUNC_CALL(bubbleSortIterative)(arr, p + 1, h, sortMode);
            } else {
                stack[++top] = p + 1;
                stack[++top] = h;
            }
        }
    }
}

inline INPUT0_TYPE FUNC(intersectionOverUnion)(const __global BoxInfo* i, const __global BoxInfo* j) {
    const INPUT0_TYPE norm = !NORMALIZED;

    INPUT0_TYPE areaI = (i->ymax - i->ymin + norm) * (i->xmax - i->xmin + norm);
    INPUT0_TYPE areaJ = (j->ymax - j->ymin + norm) * (j->xmax - j->xmin + norm);

    if (areaI <= 0.0f || areaJ <= 0.0f) { // FIXME macro
        return 0.0f;
    }

    float intersection_ymin = max(i->ymin, j->ymin);
    float intersection_xmin = max(i->xmin, j->xmin);
    float intersection_ymax = min(i->ymax, j->ymax);
    float intersection_xmax = min(i->xmax, j->xmax);

    float intersection_area = max(intersection_ymax - intersection_ymin + norm, 0.0f) *
                              max(intersection_xmax - intersection_xmin + norm, 0.0f);

    return intersection_area / (areaI + areaJ - intersection_area);
}

inline OUTPUT_INDICES_TYPE FUNC(nms)(const __global INPUT0_TYPE* boxes,
                                     const __global INPUT0_TYPE* scores,
                                     OUTPUT_INDICES_TYPE batch_idx,
                                     OUTPUT_INDICES_TYPE class_idx,
                                     __global BoxInfo* box_info) {
    size_t candidates_num = 0;

    for (OUTPUT_INDICES_TYPE box_idx = 0; box_idx < NUM_BOXES; ++box_idx) {
        if (scores[box_idx] < SCORE_THRESHOLD) {
            continue;
        }

        __global BoxInfo* candidate_box = box_info + candidates_num;
        candidate_box->class_idx = class_idx;
        candidate_box->batch_idx = batch_idx;
        candidate_box->index = box_idx;
        candidate_box->score = scores[box_idx];
        candidate_box->xmin = boxes[4 * box_idx + 0];
        candidate_box->ymin = boxes[4 * box_idx + 1];
        candidate_box->xmax = boxes[4 * box_idx + 2];
        candidate_box->ymax = boxes[4 * box_idx + 3];

        ++candidates_num;
    }

/*
    printf("OCL Before sort\n");
    for (size_t i = 0; i < candidates_num; ++i) {
        __global BoxInfo* next_candidate = box_info + i;
        printf("OCL  score %f class_idx %d batch_idx %d index %d\n", next_candidate->score, next_candidate->class_idx, next_candidate->batch_idx, next_candidate->index);
    }
*/

    // sort by score in current class - must be higher score/lower index first (std::greater<BoxInfo> in ref impl.)
    FUNC_CALL(quickSortIterative)(box_info, 0, candidates_num - 1, SORTMODE_SCORE_THEN_INDEX);

    // threshold nms_top_k for each class
    if (NMS_TOP_K > -1 && NMS_TOP_K < candidates_num) {
        candidates_num = NMS_TOP_K;
    }

    if (candidates_num <= 0) {  // early drop
        return candidates_num;  // empty
    }

/*
    printf("OCL After sort\n");
    for (size_t i = 0; i < candidates_num; ++i) {
        __global BoxInfo* next_candidate = box_info + i;
        printf("OCL  score %f class_idx %d batch_idx %d index %d\n", next_candidate->score, next_candidate->class_idx, next_candidate->batch_idx, next_candidate->index);
    }
*/
    INPUT0_TYPE adaptive_threshold = IOU_THRESHOLD;
/*
    printf("OCL after sort\n");
*/
    size_t selected_size = 0;
    for (size_t i = 0; i < candidates_num; ++i) {
        __global BoxInfo* next_candidate = box_info + i;

//        printf("next_candidate.box: %f %f %f %f\n", next_candidate->xmin, next_candidate->ymin, next_candidate->xmax, next_candidate->ymax);
//        printf("OCL  score %f class_idx %d batch_idx %d index %d\n", next_candidate->score, next_candidate->class_idx, next_candidate->batch_idx, next_candidate->index);
        bool should_hard_suppress = false;

        if (NMS_ETA < 1 && adaptive_threshold > 0.5) // FIXME: macro for half
            adaptive_threshold *= NMS_ETA;

        for (size_t j = 0; j < selected_size; ++j) {
            __global BoxInfo* selected = box_info + j;
            float iou = FUNC_CALL(intersectionOverUnion)(box_info + i, box_info + j);

//            printf("next_candidate.box: %f %f %f %f\n", next_candidate->xmin, next_candidate->ymin, next_candidate->xmax, next_candidate->ymax);
//            printf("selected.box: %f %f %f %f\n", selected->xmin, selected->ymin, selected->xmax, selected->ymax);
//            printf("  class_idx: %d, i: %d, j: %d, iou: %f\n", class_idx, i, j, iou);
            if (iou >= adaptive_threshold) {

                should_hard_suppress = true;
            }
        }
        if (!should_hard_suppress) {
            box_info[selected_size] = box_info[i];
            ++selected_size;
        }
    }


    //printf("batch_idx %d class_idx %d detection count: %d\n", (int)batch_idx, (int)class_idx, (int)selected_size);

    return selected_size;
}

inline OUTPUT_INDICES_TYPE FUNC(multiclass_nms)(const __global INPUT0_TYPE* boxes,
                                                const __global INPUT0_TYPE* scores,
                                                OUTPUT_INDICES_TYPE batch_idx,
                                                __global BoxInfo* box_info) {
    OUTPUT_INDICES_TYPE detection_count = 0;

    for (uint class_idx = 0; class_idx < NUM_CLASSES; ++class_idx) {
        if (class_idx == BACKGROUND_CLASS)
            continue;

        uint detected = FUNC_CALL(nms)(boxes, scores + class_idx * NUM_BOXES, batch_idx, class_idx, box_info + detection_count);

/*
        printf("OCL Post nms batch=%d class=%d detected=%d\n", batch_idx, class_idx, detected);
        for(uint i=0; i<detected; ++i) {
            __global const BoxInfo* box = box_info + detection_count + i;
            printf("OCL %d %d %d %f\n", box->batch_idx, box->class_idx, box->index, box->score);
        }
*/

        detection_count += detected;
    }

    FUNC_CALL(quickSortIterative)(box_info, 0, detection_count - 1, SORTMODE_SCORE_THEN_CLASS);

/*
    printf("OCL Post nms sort batch=%d \n", batch_idx);
    for(uint i=0; i<detection_count; ++i) {
        __global const BoxInfo* box = box_info + i;
        printf("OCL %d %d %d %f\n", box->batch_idx, box->class_idx, box->index, box->score);
    }
*/

/*
    if (KEEP_TOP_K > -1)
        detection_count = min(detection_count, KEEP_TOP_K);
*/

    if (KEEP_TOP_K > -1 && KEEP_TOP_K < detection_count) {
        detection_count = KEEP_TOP_K;
    }


#if !(SORT_RESULT_ACROSS_BATCH) && (SORT_RESULT_TYPE == SORT_RESULT_CLASSID)
    //printf("Oops\n");
    // lexa: still under question
    FUNC_CALL(quickSortIterative)(box_info, 0, detection_count - 1, SORTMODE_CLASS);
#endif

    //printf("batch_idx %d detection count: %d\n", (int)batch_idx, (int)detection_count);
    return detection_count;
}

KERNEL(multiclass_nms_ref)(
    const __global INPUT0_TYPE* boxes,
    const __global INPUT1_TYPE* scores,
#ifdef HAS_ROISNUM
    const __global INPUT2_TYPE* roisnum,
#endif
    __global OUTPUT_INDICES_TYPE* selected_indices,
    __global OUTPUT_INDICES_TYPE* selected_num,
    __global BoxInfo* box_info, //internal buffer
    __global OUTPUT_TYPE* selected_outputs) {

    OUTPUT_INDICES_TYPE box_info_offset = 0;

    for (uint batch_idx = 0; batch_idx < NUM_BATCHES; ++batch_idx) {
        const __global INPUT0_TYPE* boxes_ptr = boxes + batch_idx * NUM_BOXES * 4;
        const __global INPUT0_TYPE* scores_ptr = scores + batch_idx * NUM_CLASSES * NUM_BOXES;

        uint nselected = FUNC_CALL(multiclass_nms)(boxes_ptr, scores_ptr, batch_idx, box_info + box_info_offset);

        selected_num[batch_idx] = nselected;

        __global OUTPUT_TYPE* selected_outputs_ptr = selected_outputs + batch_idx * MAX_OUTPUT_BOXES_PER_BATCH * 6;
        __global OUTPUT_INDICES_TYPE* selected_indices_ptr = selected_indices + batch_idx * MAX_OUTPUT_BOXES_PER_BATCH;
        uint idx;
        for (idx = 0; idx < nselected; ++idx) {
            const __global BoxInfo* info = box_info + box_info_offset + idx;
            selected_outputs_ptr[6 * idx + 0] = (OUTPUT_TYPE)info->class_idx;
            selected_outputs_ptr[6 * idx + 1] = info->score;
            selected_outputs_ptr[6 * idx + 2] = info->xmin;
            selected_outputs_ptr[6 * idx + 3] = info->ymin;
            selected_outputs_ptr[6 * idx + 4] = info->xmax;
            selected_outputs_ptr[6 * idx + 5] = info->ymax;

            selected_indices_ptr[idx] = info->batch_idx * NUM_BOXES + info->index;
        }

        // tail
        for (; idx < MAX_OUTPUT_BOXES_PER_BATCH; ++idx) {
            selected_outputs_ptr[6 * idx + 0] = -1;
            selected_outputs_ptr[6 * idx + 1] = -1;
            selected_outputs_ptr[6 * idx + 2] = -1;
            selected_outputs_ptr[6 * idx + 3] = -1;
            selected_outputs_ptr[6 * idx + 4] = -1;
            selected_outputs_ptr[6 * idx + 5] = -1;

            selected_indices_ptr[idx] = -1;
        }

        box_info_offset += nselected;

    }// for - batch_idx

/*

    offset += selected_num[NUM_BATCHES - 1];

#    if SORT_RESULT_ACROSS_BATCH
    //printf(">>> sort across batch!!!\n");
#        if SORT_RESULT_TYPE == SORT_RESULT_SCORE
    //printf(">>> sort across batch, by score\n");
    FUNC_CALL(quickSortIterative)(box_info, 0, offset - 1, true);
#        elif SORT_RESULT_TYPE == SORT_RESULT_CLASSID
    //printf(">>> sort across batch, by class\n");
    FUNC_CALL(quickSortIterative)(box_info, 0, offset - 1, false);
#        endif
#    endif  // SORT_RESULT_ACROSS_BATCH

    //size_t output_size = min(selected_num[NUM_BATCHES - 1], OUTPUT_DIM);
//    size_t output_size = offset;

*/

    //printf("Two 2 inputs\n");
}

#else// HAS_ROISNUM

#    define INPUT_INDICES_TYPE INPUT2_TYPE

KERNEL(multiclass_nms_ref)
(const __global INPUT0_TYPE* boxes,
 const __global INPUT0_TYPE* scores,
 const __global INPUT_INDICES_TYPE* roisnum,
 __global OUTPUT_INDICES_TYPE* selected_indices,
 __global OUTPUT_INDICES_TYPE* selected_num,
 __global OUTPUT_TYPE* selected_outputs) {

#error 3 inputs is not supported at the moment
}

#endif  // HAS_ROISNUM
