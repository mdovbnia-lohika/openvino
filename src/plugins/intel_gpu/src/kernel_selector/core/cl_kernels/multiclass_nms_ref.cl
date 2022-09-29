
#include "include/batch_headers/data_types.cl"


#define SORT_RESULT_CLASSID 0
#define SORT_RESULT_SCORE 1

#define SORTMODE_CLASS 0
#define SORTMODE_SCORE 1
#define SORTMODE_SCORE_THEN_INDEX 2
#define SORTMODE_SCORE_THEN_CLASS 3

#define MAX_CANDIDATES_PER_BATCH NUM_BOXES

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

#define BoxInfo FUNC(BOX_INFO)

inline void FUNC(swap_info)(__global BoxInfo* a, __global BoxInfo* b) {
    const BoxInfo temp = *a;
    *a = *b;
    *b = temp;
}

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
            case SORTMODE_SCORE: {
                if ((arr[j].score > pivot.score) ||
                    (arr[j].score == pivot.score && arr[j].batch_idx < pivot.batch_idx) ||
                    (arr[j].score == pivot.score && arr[j].batch_idx == pivot.batch_idx &&
                     arr[j].class_idx < pivot.class_idx) ||
                    (arr[j].score == pivot.score && arr[j].batch_idx == pivot.batch_idx &&
                     arr[j].class_idx == pivot.class_idx && arr[j].index < pivot.index)) {
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
                case SORTMODE_SCORE: {
                    if ((arr[j].score > arr[j + 1].score) ||
                        (arr[j].score == arr[j + 1].score && arr[j].batch_idx < arr[j + 1].batch_idx) ||
                        (arr[j].score == arr[j + 1].score && arr[j].batch_idx == arr[j + 1].batch_idx &&
                         arr[j].class_idx < arr[j + 1].class_idx) ||
                        (arr[j].score == arr[j + 1].score && arr[j].batch_idx == arr[j + 1].batch_idx &&
                         arr[j].class_idx == arr[j + 1].class_idx && arr[j].index < arr[j + 1].index)) {
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
    if (l == h || l < 0 || h <= 0) {
        return;
    }
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
                                     uint num_boxes,
                                     __global BoxInfo* box_info) {
    size_t candidates_num = 0;

    for (OUTPUT_INDICES_TYPE box_idx = 0; box_idx < num_boxes; ++box_idx) {

        #ifdef HAS_ROISNUM
            __global INPUT0_TYPE* score_ptr = scores + class_idx * NUM_BOXES;
            __global INPUT0_TYPE* box_ptr = boxes + class_idx * NUM_BOXES * 4;
        #else
            __global INPUT0_TYPE* score_ptr = scores;
            __global INPUT0_TYPE* box_ptr = boxes;
        #endif

/*
        printf("OCL (nms) pre-score check batch_idx=%d class=%d box_idx=%d box_info=%p candidates_num=%d "
               "score_ptr=%p score=%f (%f, %f, %f, %f)\n",
               batch_idx, class_idx, box_idx, box_info, candidates_num, score_ptr, score_ptr[box_idx],
               box_ptr[4 * box_idx + 0], box_ptr[4 * box_idx + 1], box_ptr[4 * box_idx + 2], box_ptr[4 * box_idx + 3]);
*/

        if (score_ptr[box_idx] < SCORE_THRESHOLD) {
            continue;
        }

        __global BoxInfo* candidate_box = box_info + candidates_num;
        candidate_box->class_idx = class_idx;
        candidate_box->batch_idx = batch_idx;
        candidate_box->index = box_idx;
        candidate_box->score = score_ptr[box_idx];
        candidate_box->xmin = box_ptr[4 * box_idx + 0];
        candidate_box->ymin = box_ptr[4 * box_idx + 1];
        candidate_box->xmax = box_ptr[4 * box_idx + 2];
        candidate_box->ymax = box_ptr[4 * box_idx + 3];

/*
        printf("OCL (nms) batch_idx=%d candidate batch=%d class=%d box_idx=%d box_info=%p candidates_num=%d candidate_box=%p score=%f (%f, %f, %f, %f)\n",
               batch_idx, candidate_box->batch_idx, class_idx, box_idx, box_info, candidates_num, candidate_box, score_ptr[box_idx],
               candidate_box->xmin, candidate_box->ymin, candidate_box->xmax, candidate_box->ymax);
*/

        ++candidates_num;
    }

    if (candidates_num == 0) {  // early drop
        return candidates_num;
    }

/*
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (batch_idx==1) {
        printf("OCL (nms) Before sort batch_idx=%d\n", batch_idx);
        for (size_t i = 0; i < candidates_num; ++i) {
            __global BoxInfo* next_candidate = box_info + i;
            printf("OCL score %f class_idx %d batch_idx %d index %d\n", next_candidate->score, next_candidate->class_idx, next_candidate->batch_idx, next_candidate->index);
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
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
    if (batch_idx==0) {
        printf("OCL After sort\n");
        for (size_t i = 0; i < candidates_num; ++i) {
            __global BoxInfo* next_candidate = box_info + i;
            printf("OCL  score %f class_idx %d batch_idx %d index %d\n", next_candidate->score, next_candidate->class_idx, next_candidate->batch_idx, next_candidate->index);
        }
    }
*/

    INPUT0_TYPE adaptive_threshold = IOU_THRESHOLD;
/*
    printf("OCL after sort\n");
*/
    size_t selected_size = 0;
    for (size_t i = 0; i < candidates_num; ++i) {
        __global BoxInfo* next_candidate = box_info + i;

/*
        if (batch_idx == 5 && class_idx == 0) {
            printf("OCL score %f batch_idx %d class_idx %d index %d\n", next_candidate->score, next_candidate->batch_idx, next_candidate->class_idx,  next_candidate->index);
        }
*/
//        printf("next_candidate.box: %f %f %f %f\n", next_candidate->xmin, next_candidate->ymin, next_candidate->xmax, next_candidate->ymax);
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
//                printf("OCL should_hard_suppress = true score %f batch_idx %d class_idx %d index %d\n", next_candidate->score, next_candidate->batch_idx, next_candidate->class_idx,  next_candidate->index);
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
                                                const uint num_boxes,
                                                OUTPUT_INDICES_TYPE batch_idx,
                                                __global BoxInfo* box_info) {
    OUTPUT_INDICES_TYPE detection_count = 0;

    for (uint class_idx = 0; class_idx < NUM_CLASSES; ++class_idx) {
        if (class_idx == BACKGROUND_CLASS)
            continue;

        #ifdef HAS_ROISNUM
            const __global INPUT0_TYPE* boxes_ptr = boxes;
            const __global INPUT0_TYPE* scores_ptr = scores;
        #else
            const __global INPUT0_TYPE* boxes_ptr = boxes;
            const __global INPUT0_TYPE* scores_ptr = scores + class_idx * num_boxes;
        #endif

/*
        printf("OCL pre-nms batch %d class %d boxes_ptr %p scores_ptr %p box_info %p\n",
               batch_idx, class_idx, boxes_ptr, scores_ptr, box_info);
*/
        uint detected = FUNC_CALL(nms)(boxes_ptr, scores_ptr, batch_idx, class_idx, num_boxes, box_info + detection_count);

/*
        if (batch_idx == 1) {
            barrier(CLK_GLOBAL_MEM_FENCE);
            printf("OCL Post nms batch=%d class=%d detected=%d detection_count=%d\n", batch_idx, class_idx, detected, detection_count);
            for(uint i=0; i<detected; ++i) {
                __global const BoxInfo* box = box_info + detection_count + i;
                printf("OCL %d %d %d %f (%f, %f, %f, %f)\n",
                       box->batch_idx, box->class_idx, box->index, box->score,
                       box->xmin, box->ymin, box->xmax, box->ymax);
            }
        }
*/
        detection_count += detected;
    }

    FUNC_CALL(quickSortIterative)(box_info, 0, detection_count - 1, SORTMODE_SCORE_THEN_CLASS);

/*
    if (batch_idx == 1) {
        barrier(CLK_GLOBAL_MEM_FENCE);
        printf("********** detection_count=%d\n", detection_count);
        printf("OCL Post nms sort batch_idx=%d\n", batch_idx);
        for(uint i=0; i<detection_count; ++i) {
            __global const BoxInfo* box = box_info + i;
            printf("OCL %d %d %d %f (%f, %f, %f, %f)\n",
                   box->batch_idx, box->class_idx, box->index, box->score,
                   box->xmin, box->ymin, box->xmax, box->ymax);
        }
    }
*/

    if (KEEP_TOP_K > -1 && KEEP_TOP_K < detection_count) {
        detection_count = KEEP_TOP_K;
    }


#if !(SORT_RESULT_ACROSS_BATCH) && (SORT_RESULT_TYPE == SORT_RESULT_CLASSID)
    FUNC_CALL(quickSortIterative)(box_info, 0, detection_count - 1, SORTMODE_CLASS);
#endif

/*
    printf("OCL (multiclass_nms) before exit batch_idx=%d\n", batch_idx);
    for(uint i=0; i<detection_count; ++i) {
        __global const BoxInfo* box = box_info + i;
        printf("OCL %d %d %d %f (%f, %f, %f, %f)\n",
               box->batch_idx, box->class_idx, box->index, box->score,
               box->xmin, box->ymin, box->xmax, box->ymax);
    }
*/

    //printf("batch_idx %d detection count: %d\n", (int)batch_idx, (int)detection_count);
    return detection_count;
}


#ifdef MULTICLASSNMS_STAGE_0
KERNEL(multiclass_nms_ref_stage_0)(
    const __global INPUT0_TYPE* boxes,
    const __global INPUT1_TYPE* scores,
#ifdef HAS_ROISNUM
    const __global INPUT2_TYPE* roisnum,
#endif
    __global OUTPUT_INDICES_TYPE* selected_indices,
    __global OUTPUT_INDICES_TYPE* selected_num,
    __global BoxInfo* box_info, //internal buffer
    __global OUTPUT_TYPE* selected_outputs) {


/*
    printf("scores\n");
    for(uint z = 0; z < NUM_CLASSES * 100; ++z) {
        printf("%f ", scores[z]);
    }
    printf("\n");
*/
/*
    printf("boxes\n");
    for(uint z = 0; z < NUM_CLASSES * NUM_BOXES * 4; z += 4) {
        printf("(%f, %f, %f, %f)\n", boxes[z + 0], boxes[z + 1], boxes[z + 2], boxes[z + 3]);
    }
    printf("\n");
*/


    //***for (uint batch_idx = 0; batch_idx < NUM_BATCHES; ++batch_idx) {
    uint batch_idx = get_global_id(0);
    OUTPUT_INDICES_TYPE box_info_offset = batch_idx * MAX_CANDIDATES_PER_BATCH/*MAX_OUTPUT_BOXES_PER_BATCH*/;
/*
    uint boxes_offset = 0;
    uint scores_offset = 0;
    uint num_boxes;
*/

        #ifdef HAS_ROISNUM
            const uint num_boxes = roisnum[batch_idx];
            if(num_boxes <= 0) {
                selected_num[batch_idx] = 0;
                //***continue;
                return;
            }

            uint num_previous_boxes = 0;
            if (batch_idx > 0) {
                for (uint i = 0; i < batch_idx; ++i) {
                    num_previous_boxes += roisnum[i];
                }
            }

            const uint boxes_offset = num_previous_boxes * 4;
            const uint scores_offset = num_previous_boxes;

        #else
            const uint num_boxes = NUM_BOXES;
            const uint boxes_offset = batch_idx * NUM_BOXES * 4;
            const uint scores_offset = batch_idx * NUM_CLASSES * NUM_BOXES;
        #endif

        const __global INPUT0_TYPE* boxes_ptr = boxes + boxes_offset;
        const __global INPUT0_TYPE* scores_ptr = scores + scores_offset;
/*
        printf("OCL pre (multiclass_nms) batch_idx=%d box_info_offset=%d\n",
               batch_idx, box_info_offset);
        for (uint idx = 0; idx < box_info_offset; ++idx) {
            const __global BoxInfo* info = box_info + idx;
            printf("    OCL boxinfo idx=%d index=%d class_idx=%d batch_idx=%d score=%f\n",
                idx, info->index, info->class_idx, info->batch_idx, info->score);
        }
*/
//        printf("OCL main batch_idx=%d scores_offset=%d box_info_offset=%d\n", batch_idx, scores_offset, box_info_offset);
        uint nselected = FUNC_CALL(multiclass_nms)(boxes_ptr, scores_ptr, num_boxes, batch_idx, box_info + box_info_offset);
        //***box_info_offset += nselected;

        selected_num[batch_idx] = nselected;

/*
        barrier(CLK_GLOBAL_MEM_FENCE);
        if(batch_idx==0) {
            for(uint b=0; b<NUM_BATCHES; ++b) {
                box_info_offset = b * MAX_CANDIDATES_PER_BATCH;
                printf("OCL post (multiclass_nms) batch_idx=%d box_info_offset=%d nselected=%d\n",
                       b, box_info_offset, selected_num[b]);
                for (uint idx = 0; idx < selected_num[b]; ++idx) {
                    const __global BoxInfo* info = box_info + box_info_offset + idx;
                    printf("    OCL boxinfo idx=%d index=%d class_idx=%d batch_idx=%d score=%f\n",
                        idx, info->index, info->class_idx, info->batch_idx, info->score);
                }
            }
        }
*/


/*
        printf("OCL PROCESSING post multiclass_nms batch_idx=%d num_boxes=%d nselected=%d\n", batch_idx, num_boxes, nselected);
*/

        //***#ifdef HAS_ROISNUM
        //***    boxes_offset += roisnum[batch_idx] * 4;
        //***    scores_offset += roisnum[batch_idx];
        //***#endif

    //***}// for - batch_idx


/*
    printf("OCL main before sort box_info_offset=%d\n", box_info_offset);
    for (uint idx = 0; idx < box_info_offset; ++idx) {
        const __global BoxInfo* info = box_info + idx;
        printf("    OCL boxinfo idx=%d index=%d class_idx=%d batch_idx=%d score=%f\n",
            idx, info->index, info->class_idx, info->batch_idx, info->score);
    }
*/


//***#if SORT_RESULT_ACROSS_BATCH
//***    #if SORT_RESULT_TYPE == SORT_RESULT_SCORE
//***        FUNC_CALL(quickSortIterative)(box_info, 0, box_info_offset - 1, SORTMODE_SCORE);
//***    #elif SORT_RESULT_TYPE == SORT_RESULT_CLASSID
//***        FUNC_CALL(quickSortIterative)(box_info, 0, box_info_offset - 1, SORTMODE_CLASS);
//***    #endif
//***#endif  // SORT_RESULT_ACROSS_BATCH

/*
    printf("OCL main after sort box_info_offset=%d\n", box_info_offset);
    for (uint idx = 0; idx < box_info_offset; ++idx) {
        const __global BoxInfo* info = box_info + idx;
        printf("    OCL boxinfo idx=%d index=%d class_idx=%d batch_idx=%d score=%f\n",
            idx, info->index, info->class_idx, info->batch_idx, info->score);
    }
*/

//    // fill outputs
//    box_info_offset = 0;
//    for (uint batch_idx = 0; batch_idx < NUM_BATCHES; ++batch_idx) {
//        __global OUTPUT_TYPE* selected_outputs_ptr = selected_outputs + batch_idx * MAX_OUTPUT_BOXES_PER_BATCH * 6;
//        __global OUTPUT_INDICES_TYPE* selected_indices_ptr = selected_indices + batch_idx * MAX_OUTPUT_BOXES_PER_BATCH;
//
//        uint nselected = selected_num[batch_idx];
//
////        printf("OCL fill batch_idx=%d nselected=%d box_info_offset=%d\n", batch_idx, nselected, box_info_offset);
//
//        uint idx;
//        for (idx = 0; idx < nselected; ++idx) {
//            const __global BoxInfo* info = box_info + box_info_offset + idx;
//
////            printf("OCL boxinfo idx=%d index=%d class_idx=%d batch_idx=%d score=%f\n",
////                idx, info->index, info->class_idx, info->batch_idx, info->score);
//
//            selected_outputs_ptr[6 * idx + 0] = (OUTPUT_TYPE)info->class_idx;
//            selected_outputs_ptr[6 * idx + 1] = info->score;
//            selected_outputs_ptr[6 * idx + 2] = info->xmin;
//            selected_outputs_ptr[6 * idx + 3] = info->ymin;
//            selected_outputs_ptr[6 * idx + 4] = info->xmax;
//            selected_outputs_ptr[6 * idx + 5] = info->ymax;
//
//            #ifdef HAS_ROISNUM
//                uint num_boxes = roisnum[batch_idx];
//                uint offset = 0;
//                for (uint i = 0; i < info->batch_idx; ++i) {
//                    offset += roisnum[i];
//                }
//                selected_indices_ptr[idx] = (offset + info->index) * NUM_CLASSES + info->class_idx;
///*
//                printf("OCL selected_indices[idx]=%d idx=%d offset=%d index=%d class_idx=%d batch_idx=%d\n",
//                    selected_indices_ptr[idx], idx, offset, info->index, info->class_idx, info->batch_idx);
//*/
//            #else
//                selected_indices_ptr[idx] = info->batch_idx * NUM_BOXES + info->index;
//            #endif
//
//            //printf("OCL selected_indices_ptr[%d]=%d\n", idx, selected_indices_ptr[idx]);
//        }
//
////        printf("OCL before tail batch_idx=%d idx=%d nselected=%d\n", batch_idx, idx, nselected);
//        // tail
//        for (; idx < MAX_OUTPUT_BOXES_PER_BATCH; ++idx) {
//            selected_outputs_ptr[6 * idx + 0] = -1;
//            selected_outputs_ptr[6 * idx + 1] = -1;
//            selected_outputs_ptr[6 * idx + 2] = -1;
//            selected_outputs_ptr[6 * idx + 3] = -1;
//            selected_outputs_ptr[6 * idx + 4] = -1;
//            selected_outputs_ptr[6 * idx + 5] = -1;
//
//            selected_indices_ptr[idx] = -1;
//        }
//
//        box_info_offset += nselected;
//    }

/*
    printf("OCL selected_indices:\n");
    for(uint b=0; b < NUM_BATCHES; ++b) {
        for(uint z=0; z < MAX_OUTPUT_BOXES_PER_BATCH; ++z) {
            printf("%d ", selected_indices[b * MAX_OUTPUT_BOXES_PER_BATCH + z]);
        }
        printf("\n");
    }
*/
}

#endif //MULTICLASSNMS_STAGE_0

#ifdef MULTICLASSNMS_STAGE_1
KERNEL(multiclass_nms_ref_stage_1)(
    const __global INPUT0_TYPE* boxes,
    const __global INPUT1_TYPE* scores,
#ifdef HAS_ROISNUM
    const __global INPUT2_TYPE* roisnum,
#endif
    __global OUTPUT_INDICES_TYPE* selected_indices,
    __global OUTPUT_INDICES_TYPE* selected_num,
    __global BoxInfo* box_info, //internal buffer
    __global OUTPUT_TYPE* selected_outputs) {

    // pack box_infos
    uint dst_offset = selected_num[0];
    for(uint batch_idx = 0; batch_idx < NUM_BATCHES-1; ++batch_idx) {
        uint boxes_to_copy = selected_num[batch_idx+1];
        uint src_offset = (batch_idx+1) * MAX_CANDIDATES_PER_BATCH/*MAX_OUTPUT_BOXES_PER_BATCH*/;

        for(uint i = 0; i < boxes_to_copy; ++i) {
            box_info[dst_offset + i] = box_info[src_offset + i];
        }

        dst_offset += boxes_to_copy;
    }

#if SORT_RESULT_ACROSS_BATCH
    #if SORT_RESULT_TYPE == SORT_RESULT_SCORE
        FUNC_CALL(quickSortIterative)(box_info, 0, dst_offset - 1, SORTMODE_SCORE);
    #elif SORT_RESULT_TYPE == SORT_RESULT_CLASSID
        FUNC_CALL(quickSortIterative)(box_info, 0, dst_offset - 1, SORTMODE_CLASS);
    #endif
#endif  // SORT_RESULT_ACROSS_BATCH
}
#endif //MULTICLASSNMS_STAGE_1

#ifdef MULTICLASSNMS_STAGE_2
KERNEL(multiclass_nms_ref_stage_2)(
    const __global INPUT0_TYPE* boxes,
    const __global INPUT1_TYPE* scores,
#ifdef HAS_ROISNUM
    const __global INPUT2_TYPE* roisnum,
#endif
    __global OUTPUT_INDICES_TYPE* selected_indices,
    __global OUTPUT_INDICES_TYPE* selected_num,
    __global BoxInfo* box_info, //internal buffer
    __global OUTPUT_TYPE* selected_outputs) {

    // fill outputs
    //box_info_offset = 0;

    uint batch_idx = get_global_id(0);
    //for (uint batch_idx = 0; batch_idx < NUM_BATCHES; ++batch_idx) {
        __global OUTPUT_TYPE* selected_outputs_ptr = selected_outputs + batch_idx * MAX_OUTPUT_BOXES_PER_BATCH * 6;
        __global OUTPUT_INDICES_TYPE* selected_indices_ptr = selected_indices + batch_idx * MAX_OUTPUT_BOXES_PER_BATCH;

        uint nselected = selected_num[batch_idx];

        uint box_info_offset = 0;
        for (uint i = 0; i < batch_idx; ++i) {
            box_info_offset += selected_num[i];
        }

        //printf("OCL fill batch_idx=%d nselected=%d box_info_offset=%d\n", batch_idx, nselected, box_info_offset);

        uint idx;
        for (idx = 0; idx < nselected; ++idx) {
            const __global BoxInfo* info = box_info + box_info_offset + idx;

            selected_outputs_ptr[6 * idx + 0] = (OUTPUT_TYPE)info->class_idx;
            selected_outputs_ptr[6 * idx + 1] = info->score;
            selected_outputs_ptr[6 * idx + 2] = info->xmin;
            selected_outputs_ptr[6 * idx + 3] = info->ymin;
            selected_outputs_ptr[6 * idx + 4] = info->xmax;
            selected_outputs_ptr[6 * idx + 5] = info->ymax;

            #ifdef HAS_ROISNUM
                uint num_boxes = roisnum[batch_idx];
                uint offset = 0;
                for (uint i = 0; i < info->batch_idx; ++i) {
                    offset += roisnum[i];
                }
                selected_indices_ptr[idx] = (offset + info->index) * NUM_CLASSES + info->class_idx;

/*
                printf("OCL selected_indices[idx]=%d batch_idx=%d idx=%d offset=%d index=%d class_idx=%d batch_idx=%d\n",
                    selected_indices_ptr[idx], batch_idx, idx, offset, info->index, info->class_idx, info->batch_idx);
*/

            #else
                selected_indices_ptr[idx] = info->batch_idx * NUM_BOXES + info->index;
            #endif

/*
            printf("OCL boxinfo batch_idx=%d selected_indices_ptr[idx]=%d idx=%d box_info_offset=%d index=%d class_idx=%d batch_idx=%d score=%f\n",
                batch_idx, selected_indices_ptr[idx], idx, box_info_offset, info->index, info->class_idx, info->batch_idx, info->score);
*/

        }

//        printf("OCL before tail batch_idx=%d idx=%d nselected=%d\n", batch_idx, idx, nselected);
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
    //}

/*
    printf("OCL selected_indices:\n");
    for(uint b=0; b < NUM_BATCHES; ++b) {
        for(uint z=0; z < MAX_OUTPUT_BOXES_PER_BATCH; ++z) {
            printf("%d ", selected_indices[b * MAX_OUTPUT_BOXES_PER_BATCH + z]);
        }
        printf("\n");
    }
*/
}
#endif //MULTICLASSNMS_STAGE_2
