// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/multiclass_nms.hpp>
#include <intel_gpu/primitives/mutable_data.hpp>

#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace {

template <typename T>
float getError();

 template <>
 float getError<FLOAT16>() {
     return 0.2;
 }

template <>
float getError<float>() {
    return 0.001;
}

 template <>
 float getError<half_t>() {
     return 0.2;
 }

template <typename T, typename T_IND>
struct multiclass_nms_test_input {
    cldnn::sort_result_type sort_result_type;
    bool sort_result_across_batch;
    float iou_threshold;
    float score_threshold;
    int nms_top_k;
    int keep_top_k;
    int background_class;
    bool normalized;
    float nms_eta;
    bool has_rois_num;

    size_t num_batches;
    size_t num_classes;
    size_t num_boxes;

    std::vector<T> boxes;
    std::vector<T> scores;
    std::vector<T_IND> rois_num;

    std::vector<float> expected_selected_outputs;
    std::vector<T_IND> expected_selected_indices;
    std::vector<T_IND> expected_selected_num;
    std::string test_name;
};

template <typename T, typename T_IND>
using multiclass_nms_test_params = std::tuple<multiclass_nms_test_input<T, T_IND>, format::type>;

template <typename T, typename T_IND>
struct multiclass_nms_test : public ::testing::TestWithParam<multiclass_nms_test_params<T, T_IND>> {
public:
    void test() {
        multiclass_nms_test_input<T, T_IND> param;
        format::type input_format;
        std::tie(param, input_format) = testing::TestWithParam<multiclass_nms_test_params<T, T_IND>>::GetParam();
        auto data_type = type_to_data_type<T>::value;
        auto index_data_type = type_to_data_type<T_IND>::value;
        constexpr auto plain_format = format::bfyx;

        convert<T>(param.expected_selected_outputs);

        auto& engine = get_test_engine();

        const auto input_boxes =
            !param.has_rois_num
                ? engine.allocate_memory({data_type,
                                          plain_format,
                                          tensor{batch(param.num_batches), feature(param.num_boxes), spatial(1, 4)}})
                : engine.allocate_memory({data_type,
                                          plain_format,
                                          tensor{batch(param.num_classes), feature(param.num_boxes), spatial(1, 4)}});
        set_values(input_boxes, param.boxes);

        auto input_scores =
            !param.has_rois_num
                ? engine.allocate_memory(
                      {data_type,
                       plain_format,
                       tensor{batch(param.num_batches), feature(param.num_classes), spatial(1, param.num_boxes)}})
                : engine.allocate_memory(
                      {data_type, plain_format, tensor{batch(param.num_classes), feature(param.num_boxes)}});
        set_values(input_scores, param.scores);

        auto input_rois_num =
            param.has_rois_num
                ? engine.allocate_memory({index_data_type, plain_format, tensor{batch(param.num_batches)}})
                : nullptr;
        if (input_rois_num)
            set_values(input_rois_num, param.rois_num);

        // calculate output static dim {
        auto real_num_classes = param.num_classes;
        if (param.background_class >= 0 && static_cast<uint>(param.background_class) < param.num_classes) {
            real_num_classes = std::max(1ul, param.num_classes - 1);
        }
        int64_t max_output_boxes_per_class = 0;
        if (param.nms_top_k >= 0)
            max_output_boxes_per_class = std::min((int)param.num_boxes, param.nms_top_k);
        else
            max_output_boxes_per_class = param.num_boxes;

        auto max_output_boxes_per_batch = max_output_boxes_per_class * real_num_classes;
        if (param.keep_top_k >= 0)
            max_output_boxes_per_batch = std::min((int)max_output_boxes_per_batch, param.keep_top_k);

        const auto dim = max_output_boxes_per_batch * param.num_batches;
        // } end calculate

        const layout output_selected_indices_layout{index_data_type, input_format, tensor{batch(dim), feature(1)}};
        auto output_selected_indices = engine.allocate_memory(output_selected_indices_layout);
        const layout output_selected_num_layout{index_data_type, input_format, tensor{batch(param.num_batches)}};
        auto output_selected_num = engine.allocate_memory(output_selected_num_layout);

        topology topology;

        topology.add(input_layout("input_boxes", input_boxes->get_layout()));
        topology.add(input_layout("input_scores", input_scores->get_layout()));
        if (param.has_rois_num) {
            topology.add(input_layout("input_rois_num", input_rois_num->get_layout()));
        }

        topology.add(mutable_data("output_selected_indices", output_selected_indices));
        topology.add(mutable_data("output_selected_num", output_selected_num));

        topology.add(reorder("input_boxes_reordered", "input_boxes", input_format, data_type));
        topology.add(reorder("input_scores_reordered", "input_scores", input_format, data_type));
        if (param.has_rois_num) {
            topology.add(reorder("input_rois_num_reordered", "input_rois_num", input_format, index_data_type));
        }

        const auto primitive = multiclass_nms{
            "multiclass_nms_reordered",
            "input_boxes_reordered",
            "input_scores_reordered",
            param.has_rois_num ? "input_rois_num_reordered" : "",
            "output_selected_indices",
            "output_selected_num",
            param.sort_result_type,
            param.sort_result_across_batch,
            index_data_type,
            param.iou_threshold,
            param.score_threshold,
            param.nms_top_k,
            param.keep_top_k,
            param.background_class,
            param.normalized,
            param.nms_eta,
        };

        topology.add(primitive);
        topology.add(reorder("multiclass_nms", "multiclass_nms_reordered", plain_format, data_type));
        build_options build_opt;
        build_opt.set_option(build_option::optimize_data(false));
        network network(engine, topology, build_opt);

        network.set_input_data("input_boxes", input_boxes);
        network.set_input_data("input_scores", input_scores);
        if (param.has_rois_num)
            network.set_input_data("input_rois_num", input_rois_num);

        const auto outputs = network.execute();

        const auto output_boxes = outputs.at("multiclass_nms").get_memory();

        const cldnn::mem_lock<T> output_boxes_ptr(output_boxes, get_test_stream());
        ASSERT_EQ(output_boxes_ptr.size(), dim * 6);

        const auto get_plane_data = [&](const memory::ptr& mem, const data_types data_type, const layout& from_layout) {
            if (from_layout.format == plain_format) {
                return mem;
            }
            cldnn::topology reorder_topology;
            reorder_topology.add(input_layout("data", from_layout));
            reorder_topology.add(reorder("plane_data", "data", plain_format, data_type));
            cldnn::network reorder_net{engine, reorder_topology};
            reorder_net.set_input_data("data", mem);
            const auto second_output_result = reorder_net.execute();
            const auto plane_data_mem = second_output_result.at("plane_data").get_memory();
            return plane_data_mem;
        };

        const cldnn::mem_lock<T_IND> output_selected_indices_ptr(
            get_plane_data(output_selected_indices, index_data_type, output_selected_indices_layout),
            get_test_stream());
        ASSERT_EQ(output_selected_indices_ptr.size(), dim);

        const cldnn::mem_lock<T_IND> output_selected_num_ptr(
            get_plane_data(output_selected_num, index_data_type, output_selected_num_layout),
            get_test_stream());
        ASSERT_EQ(output_selected_num_ptr.size(), param.num_batches);

        for (size_t i = 0; i < param.num_batches; ++i) {
            EXPECT_EQ(param.expected_selected_num[i], output_selected_num_ptr[i]);
        }

        for (size_t i = 0; i < dim; ++i) {
            for (size_t j = 0; j < 6; ++j) {
                const auto idx = i * 6 + j;
                EXPECT_NEAR(param.expected_selected_outputs[idx], output_boxes_ptr[idx], getError<T>());
            }
            EXPECT_EQ(param.expected_selected_indices[i], output_selected_indices_ptr[i]);
        }
    }
    static std::string PrintToStringParamName(
        const testing::TestParamInfo<multiclass_nms_test_params<T, T_IND>>& info) {
        auto& p = std::get<0>(info.param);
        std::ostringstream result;
        result << "NumBatches" << p.num_batches << "_";
        result << "InputType=" << data_type_traits::name(type_to_data_type<T_IND>::value) << "_";
        result << "DataType=" << data_type_traits::name(type_to_data_type<T>::value) << "_";
        result << "Format=" << fmt_to_str(std::get<1>(info.param));
        return result.str();
    }
};

const std::vector<format::type> layout_formats = {
    format::bfyx,
                                                      format::b_fs_yx_fsv16,
                                                      format::b_fs_yx_fsv32,
                                                      format::bs_fs_yx_bsv16_fsv16,
                                                      format::bs_fs_yx_bsv32_fsv16,
                                                      format::bs_fs_yx_bsv32_fsv32
};

template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_nms_score_input() {
    return {
        {cldnn::sort_result_type::score,  // sort_result_type
         false,                           // sort_result_across_batch
         0.5f,                            // iou_threshold
         0.0f,                            // score_threshold
         3,                               // nms_top_k
         -1,                              // keep_top_k
         -1,                              // background_class
         true,                            // normalized
         1.0f,                            // nms_eta
         false,                           // has_rois_num

         1,  // num_batches
         2,  // num_classes
         6,  // num_boxes

         convert<T>({0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                     0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),  // boxes
         convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),     // scores
         std::vector<T_IND>{},                                                              // rois_num
         std::vector<float>{
             0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 1.00, 0.95, 0.00, 0.00,  1.00, 1.00,  // expected_selected_outputs
             0.00, 0.90, 0.00, 0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00,
             -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0},
         std::vector<T_IND>{3, 0, 0, 3, -1, -1},  // expected_selected_indices
         std::vector<T_IND>{4},                   // expected_selected_num
         "multiclass_nms_by_score"}};             // test_name
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_nms_by_class_id_input() {
    return {{cldnn::sort_result_type::classid,
             false,
             0.5f,
             0.0f,
             3,
             -1,
             -1,
             true,
             1.0f,
             false,

             1,
             2,
             6,

             convert<T>({0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                         0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
             convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),
             std::vector<T_IND>{},

             std::vector<float>{0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 0.00, 0.00,  1.00, 1.00,
                                1.00, 0.95, 0.00, 0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00,
                                -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0},

             std::vector<T_IND>{3, 0, 0, 3, -1, -1},
             std::vector<T_IND>{4},
             "multiclass_nms_by_class_id"}};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_nms_output_type_i32_input() {
    return {{cldnn::sort_result_type::score,
             false,
             0.5f,
             0.0f,
             3,
             -1,
             -1,
             true,
             1.0f,
             false,

             2,
             2,
             6,

             convert<T>({0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,   0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,  0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
             convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                         0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),
             std::vector<T_IND>{},

             std::vector<float>{
                 0.00,  0.95, 0.00,  10.00, 1.00, 11.00, 1.00,  0.95, 0.00,  0.00, 1.00, 1.00, 0.00,  0.90, 0.00,
                 0.00,  1.00, 1.00,  1.00,  0.80, 0.00,  10.00, 1.00, 11.00, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,
                 -1.0,  -1.0, -1.0,  -1.0,  -1.0, -1.0,  1.00,  0.95, 0.00,  0.00, 1.00, 1.00, 1.0,   0.80, 0.00,
                 10.00, 1.00, 11.00, 1.0,   0.5,  0.0,   10.1,  1.0,  11.1,  1.0,  0.3,  0.0,  100.0, 1.0,  101.0,
                 -1.0,  -1.0, -1.0,  -1.0,  -1.0, -1.0,  -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0,
             },
             std::vector<T_IND>{3, 0, 0, 3, -1, -1, 6, 9, 4, 5, -1, -1},
             std::vector<T_IND>{4, 4},
             "multiclass_nms_output_type_i32"}};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_nms_two_batches_two_classes_by_score_input() {
    return {{cldnn::sort_result_type::classid,
             false,
             0.5f,
             0.0f,
             3,
             -1,
             -1,
             true,
             1.0f,
             false,

             2,
             2,
             6,

             convert<T>({0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,   0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,  0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
             convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                         0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),
             std::vector<T_IND>{},

             std::vector<float>{
                 0.00,  0.95, 0.00,   10.00, 1.00, 11.00, 0.00,  0.90, 0.00,  0.00, 1.00, 1.00, 1.00,  0.95, 0.00,
                 0.00,  1.00, 1.00,   1.00,  0.80, 0.00,  10.00, 1.00, 11.00, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,
                 -1.0,  -1.0, -1.0,   -1.0,  -1.0, -1.0,  1.00,  0.5,  0,     10.1, 1.00, 11.1, 1.00,  0.3,  0.00,
                 100.0, 1.00, 101.00, 1.00,  0.95, 0.00,  0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00,
                 -1.0,  -1.0, -1.0,   -1.0,  -1.0, -1.0,  -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0,
             },
             std::vector<T_IND>{3, 0, 0, 3, -1, -1, 4, 5, 6, 9, -1, -1},
             std::vector<T_IND>{4, 4},
             "multiclass_nms_two_batches_two_classes_by_score"}};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_nms_two_batches_two_classes_by_class_id_input() {
    return {{cldnn::sort_result_type::score,
             true,
             0.5f,
             0.0f,
             3,
             -1,
             -1,
             true,
             1.0f,
             false,

             2,
             2,
             6,

             convert<T>({0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,   0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,  0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
             convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                         0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),
             std::vector<T_IND>{},

             std::vector<float>{0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 1.00, 0.95, 0.00, 0.00,  1.00, 1.00,
                                1.00, 0.95, 0.00, 0.00,  1.00, 1.00,  0.00, 0.90, 0.00, 0.00,  1.00, 1.00,
                                -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,
                                1.00, 0.80, 0.00, 10.00, 1.00, 11.00, 1.00, 0.80, 0.00, 10.00, 1.00, 11.00,
                                1.0,  0.5,  0.0,  10.1,  1.0,  11.1,  1.0,  0.3,  0.0,  100.0, 1.0,  101.0,
                                -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0},
             std::vector<T_IND>{3, 0, 6, 0, -1, -1, 3, 9, 4, 5, -1, -1},
             std::vector<T_IND>{4, 4},
             "multiclass_nms_two_batches_two_classes_by_class_id"}};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>>
getMulticlass_nms_two_batches_two_classes_by_score_cross_batch_input() {
    return {{cldnn::sort_result_type::classid,
             true,
             0.5f,
             0.0f,
             3,
             -1,
             -1,
             true,
             1.0f,
             false,

             2,
             2,
             6,

             convert<T>({0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,   0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,  0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
             convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                         0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),
             std::vector<T_IND>{},

             std::vector<float>{0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 0.00, 0.00,  1.00, 1.00,
                                1.00, 0.95, 0.00, 0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00,
                                -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,

                                1.0,  0.5,  0.0,  10.1,  1.0,  11.1,  1.0,  0.3,  0.0,  100.0, 1.0,  101.0,
                                1.00, 0.95, 0.00, 0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00,
                                -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0},
             std::vector<T_IND>{3, 0, 0, 3, -1, -1, 4, 5, 6, 9, -1, -1},
             std::vector<T_IND>{4, 4},
             "multiclass_nms_two_batches_two_classes_by_score_cross_batch"}};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_nms_flipped_coordinates_input() {
    return {{cldnn::sort_result_type::score,
             false,
             0.5f,
             0.0f,
             3,
             -1,
             -1,
             true,
             1.0f,
             false,

             1,
             1,
             6,

             convert<T>({1.0, 1.0,  0.0, 0.0,  0.0, 0.1,  1.0, 1.1,  0.0, 0.9,   1.0, -0.1,
                         0.0, 10.0, 1.0, 11.0, 1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0}),
             convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),
             std::vector<T_IND>{},

             std::vector<float>{0.00,
                                0.95,
                                0.00,
                                10.00,
                                1.00,
                                11.00,
                                0.00,
                                0.90,
                                1.00,
                                1.00,
                                0.00,
                                0.00,
                                0.00,
                                0.75,
                                0.00,
                                0.10,
                                1.00,
                                1.10},
             std::vector<T_IND>{3, 0, 1},
             std::vector<T_IND>{3},
             "multiclass_nms_flipped_coordinates"}};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_nms_identical_boxes_input() {
    return {{cldnn::sort_result_type::score,
             false,
             0.5f,
             0.0f,
             3,
             -1,
             -1,
             true,
             1.0f,
             false,

             1,
             1,
             10,

             convert<T>({0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
                         1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                         0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0}),
             convert<T>({0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9}),
             std::vector<T_IND>{},

             std::vector<float>{
                 0.00,
                 0.90,
                 0.00,
                 0.00,
                 1.00,
                 1.00,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
             },
             std::vector<T_IND>{0, -1, -1},
             std::vector<T_IND>{1},
             "multiclass_nms_identical_boxes"}};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_limit_output_size_input() {
    return {{cldnn::sort_result_type::score,
             false,
             0.5f,
             0.0f,
             2,
             -1,
             -1,
             true,
             1.0f,
             false,

             1,
             1,
             6,

             convert<T>({0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                         0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
             convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),
             std::vector<T_IND>{},

             std::vector<float>{
                 0.00,
                 0.95,
                 0.00,
                 10.00,
                 1.00,
                 11.00,
                 0.00,
                 0.90,
                 0.00,
                 0.00,
                 1.00,
                 1.00,
             },
             std::vector<T_IND>{3, 0},
             std::vector<T_IND>{2},
             "multiclass_nms_limit_output_size"}};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_single_box_input() {
    return {{cldnn::sort_result_type::score,
             false,
             0.5f,
             0.0f,
             3,
             -1,
             -1,
             true,
             1.0f,
             false,

             1,
             1,
             1,

             convert<T>({0.0, 0.0, 1.0, 1.0}),
             convert<T>({0.9}),
             std::vector<T_IND>{},

             std::vector<float>{
                 0.00,
                 0.90,
                 0.00,
                 0.00,
                 1.00,
                 1.00,
             },
             std::vector<T_IND>{0},
             std::vector<T_IND>{1},
             "multiclass_nms_single_box"}};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_IOU_input() {
    return {{cldnn::sort_result_type::score,
             false,
             0.2f,
             0.0f,
             3,
             -1,
             -1,
             true,
             1.0f,
             false,

             1,
             1,
             6,

             convert<T>({0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                         0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
             convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),
             std::vector<T_IND>{},

             std::vector<float>{
                 0.00,
                 0.95,
                 0.00,
                 10.00,
                 1.00,
                 11.00,
                 0.00,
                 0.90,
                 0.00,
                 0.00,
                 1.00,
                 1.00,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
             },
             std::vector<T_IND>{3, 0, -1},
             std::vector<T_IND>{2},
             "multiclass_nms_by_IOU"}};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_IOU_and_scores_input() {
    return {{cldnn::sort_result_type::score,
             false,
             0.5f,
             0.95f,
             3,
             -1,
             -1,
             true,
             1.0f,
             false,

             1,
             1,
             6,

             convert<T>({0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                         0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
             convert<T>({0.9, 0.75, 0.6, 0.96, 0.5, 0.3}),
             std::vector<T_IND>{},

             std::vector<float>{
                 0.00,
                 0.96,
                 0.00,
                 10.00,
                 1.00,
                 11.00,

                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
             },
             std::vector<T_IND>{3, -1, -1},
             std::vector<T_IND>{1},
             "multiclass_nms_by_IOU_and_scores"}};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_no_output_input() {
    return {
        {cldnn::sort_result_type::score,
         false,
         0.5f,
         2.0f,
         3,
         -1,
         -1,
         true,
         1.0f,
         false,

         1,
         1,
         6,

         convert<T>({0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,
                     0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
         convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),
         std::vector<T_IND>{},

         std::vector<float>{
             -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
             -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
         },
         std::vector<T_IND>{-1, -1, -1, -1, -1, -1},
         std::vector<T_IND>{0},
         "multiclass_nms_no_output"}};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_background_input() {
    return {{cldnn::sort_result_type::classid,
             false,
             0.5f,
             0.0f,
             3,
             -1,
             0,
             true,
             1.0f,
             false,

             2,
             2,
             6,

             convert<T>({0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,   0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,  0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
             convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                         0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),
             std::vector<T_IND>{},

             std::vector<float>{
                 1.00, 0.95, 0.00, 0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00,
                 -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,  1.00, 0.95, 0.00, 0.00,  1.00, 1.00,
                 1.00, 0.80, 0.00, 10.00, 1.00, 11.00, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,
             },
             std::vector<T_IND>{0, 3, -1, 6, 9, -1},
             std::vector<T_IND>{2, 2},
             "multiclass_nms_by_background"}};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_keep_top_k_input() {
    return {{cldnn::sort_result_type::classid,
             false,
             0.5f,
             0.0f,
             3,
             3,
             -1,  // background class
             true,
             1.0f,
             false,

             2,
             2,
             6,

             convert<T>({0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,   0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,  0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
             convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                         0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),
             std::vector<T_IND>{},

             std::vector<float>{
                 0.00, 0.95, 0.00, 10.00, 1.00, 11.00, 0.00, 0.90, 0.00, 0.00,  1.00, 1.00,
                 1.00, 0.95, 0.00, 0.00,  1.00, 1.00,  1.00, 0.5,  0.00, 10.1,  1.00, 11.1,
                 1.00, 0.95, 0.00, 0.00,  1.00, 1.00,  1.00, 0.80, 0.00, 10.00, 1.00, 11.00,
             },
             std::vector<T_IND>{3, 0, 0, 4, 6, 9},
             std::vector<T_IND>{3, 3},
             "multiclass_nms_by_keep_top_k"}};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_keep_post_threshold_inputs() {
    return {{cldnn::sort_result_type::classid,
             false,
             0.5f,
             0.0f,
             3,
             3,
             -1,  // background class
             true,
             1.0f,
             false,

             1,
             1,
             6,

             convert<T>({0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,  // boxes
                         0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
             convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),
             std::vector<T_IND>{},

             std::vector<float>{
                 0.00,
                 0.95,
                 0.00,
                 10.00,
                 1.00,
                 11.00,
                 0.00,
                 0.9,
                 0.00,
                 0.00,
                 1.00,
                 1.00,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
                 -1.0,
             },
             std::vector<T_IND>{3, 0, -1},
             std::vector<T_IND>{2},
             "multiclass_nms_post_threshold"}};
}

template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_eta_input() {
    return {{cldnn::sort_result_type::classid,
             false,
             1.0f,
             0.0f,
             -1,
             -1,
             -1,  // background class
             true,
             0.1f,
             false,

             2,
             2,
             6,

             convert<T>({0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,   0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                         0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0, 0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                         0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,  0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
             convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3,
                         0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),
             std::vector<T_IND>{},

             std::vector<float>{
                 0.00,   0.95, 0.00,   10.00,  1.00, 11.00,  0.00, 0.90, 0.00, 0.00, 1.00, 1.00, 0.00,  0.30, 0.00,
                 100.00, 1.00, 101.00, 1.00,   0.95, 0.00,   0.00, 1.00, 1.00, 1.00, 0.80, 0.00, 10.00, 1.00, 11.00,
                 1.00,   0.30, 0.00,   100.00, 1.00, 101.00, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,
                 -1.0,   -1.0, -1.0,   -1.0,   -1.0, -1.0,   -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,
                 -1.0,   -1.0, -1.0,   -1.0,   -1.0, -1.0,   -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,

                 1.0,    0.6,  0.0,    -0.1,   1.0,  0.9,    1.0,  0.5,  0.0,  10.1, 1.0,  11.1, 1.00,  0.30, 0.00,
                 100.00, 1.00, 101.00, 1.00,   0.95, 0.00,   0.00, 1.00, 1.00, 1.00, 0.80, 0.00, 10.00, 1.00, 11.00,
                 1.00,   0.30, 0.00,   100.00, 1.00, 101.00, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,
                 -1.0,   -1.0, -1.0,   -1.0,   -1.0, -1.0,   -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  -1.0, -1.0,
                 -1.0,   -1.0, -1.0,   -1.0,   -1.0, -1.0,   -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
             },
             std::vector<T_IND>{3, 0, 5, 0, 3, 5, -1, -1, -1, -1, -1, -1, 2, 4, 5, 6, 9, 11, -1, -1, -1, -1, -1, -1},
             std::vector<T_IND>{6, 6},
             "multiclass_nms_by_nms_eta"}};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_nms_three_inputs_blocked_layout() {
    multiclass_nms_test_input<T, T_IND> param = {
        cldnn::sort_result_type::score,
        false,
        0.5f,
        0.0f,
        3,
        -1,
        -1,
        true,
        1.0f,
        true,

        34,  // batches
        2,   // classes
        6,   // boxes

        convert<T>({
            0.0, 0.0,   1.0, 1.0,   0.0, 0.1,   1.0, 1.1,   0.0, -0.1,  1.0, 0.9,   0.0, 10.0,  1.0, 11.0,  0.0, 10.1,
            1.0, 11.1,  0.0, 100.0, 1.0, 101.0, 0.0, 0.0,   1.0, 1.0,   0.0, 0.1,   1.0, 1.1,   0.0, -0.1,  1.0, 0.9,
            0.0, 10.0,  1.0, 11.0,  0.0, 10.1,  1.0, 11.1,  0.0, 100.0, 1.0, 101.0, 0.0, 0.0,   1.0, 1.0,   0.0, 0.1,
            1.0, 1.1,   0.0, -0.1,  1.0, 0.9,   0.0, 10.0,  1.0, 11.0,  0.0, 10.1,  1.0, 11.1,  0.0, 100.0, 1.0, 101.0,
            0.0, 0.0,   1.0, 1.0,   0.0, 0.1,   1.0, 1.1,   0.0, -0.1,  1.0, 0.9,   0.0, 10.0,  1.0, 11.0,  0.0, 10.1,
            1.0, 11.1,  0.0, 100.0, 1.0, 101.0, 0.0, 0.0,   1.0, 1.0,   0.0, 0.1,   1.0, 1.1,   0.0, -0.1,  1.0, 0.9,
            0.0, 10.0,  1.0, 11.0,  0.0, 0.0,   1.0, 1.0,   0.0, 0.1,   1.0, 1.1,   0.0, -0.1,  1.0, 0.9,   0.0, 10.0,
            1.0, 11.0,  0.0, 0.0,   1.0, 1.0,   0.0, 0.1,   1.0, 1.1,   0.0, -0.1,  1.0, 0.9,   0.0, 10.0,  1.0, 11.0,
            0.0, 10.1,  1.0, 11.1,  0.0, 100.0, 1.0, 101.0, 0.0, 0.0,   1.0, 1.0,   0.0, 0.1,   1.0, 1.1,   0.0, -0.1,
            1.0, 0.9,   0.0, 10.0,  1.0, 11.0,  0.0, 10.1,  1.0, 11.1,  0.0, 100.0, 1.0, 101.0, 0.0, 0.0,   1.0, 1.0,
            0.0, 0.1,   1.0, 1.1,   0.0, -0.1,  1.0, 0.9,   0.0, 10.0,  1.0, 11.0,  0.0, 10.1,  1.0, 11.1,  0.0, 100.0,
            1.0, 101.0, 0.0, 0.0,   1.0, 1.0,   0.0, 0.1,   1.0, 1.1,   0.0, -0.1,  1.0, 0.9,   0.0, 10.0,  1.0, 11.0,
            0.0, 10.1,  1.0, 11.1,  0.0, 100.0, 1.0, 101.0, 0.0, 0.0,   1.0, 1.0,   0.0, 0.1,   1.0, 1.1,   0.0, -0.1,
            1.0, 0.9,   0.0, 10.0,  1.0, 11.0,  0.0, 0.0,   1.0, 1.0,   0.0, 0.1,   1.0, 1.1,   0.0, -0.1,  1.0, 0.9,
            0.0, 10.0,  1.0, 11.0,  0.0, 0.0,   1.0, 1.0,   0.0, 0.1,   1.0, 1.1,   0.0, -0.1,  1.0, 0.9,   0.0, 10.0,
            1.0, 11.0,
        }),
        convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.95, 0.75, 0.6, 0.80, 0.5, 0.3}),
        std::vector<T_IND>{1, 1},

        std::vector<float>({
            1.00, 0.95, 0.00, 0.00, 1.00, 1.00, 0.00, 0.90, 0.00, 0.00, 1.00, 1.00, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            0.0,  0.75, 0.0,  0.1,  1.0,  1.1,  1.0,  0.75, 0.0,  0.1,  1.0,  1.1,  -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        }),
        std::vector<T_IND>{1, 0, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1},
        std::vector<T_IND>{2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        "three_inputs_blocked_layout"};

    const auto indices_size = param.num_batches * param.num_boxes;
    const auto filled_indices = param.expected_selected_indices.size();
    param.expected_selected_indices.resize(indices_size);
    for (auto i = filled_indices; i < indices_size; ++i) {
        param.expected_selected_indices[i] = -1;
    }

    const auto outputs_size = param.num_batches * param.num_classes * param.num_boxes * 6;
    const auto filled_outputs = param.expected_selected_outputs.size();
    param.expected_selected_outputs.resize(outputs_size);
    for (auto i = filled_outputs; i < outputs_size; ++i) {
        param.expected_selected_outputs[i] = -1.0;
    }

    return {param};
}
template <typename T, typename T_IND>
std::vector<multiclass_nms_test_input<T, T_IND>> getMulticlass_keep_three_inputs() {
    multiclass_nms_test_input<T, T_IND> param = {
        cldnn::sort_result_type::score,
        false,
        0.5f,
        0.0f,
        3,
        3,
        -1,
        true,
        1.0f,
        true,

        1,
        1,
        6,

        convert<T>({0.0, 0.0,  1.0, 1.0,  0.0, 0.1,  1.0, 1.1,  0.0, -0.1,  1.0, 0.9,  // boxes
                    0.0, 10.0, 1.0, 11.0, 0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0}),
        convert<T>({0.9, 0.75, 0.6, 0.95, 0.5, 0.3}),
        std::vector<T_IND>{2, 2},

        std::vector<float>{
            0,
            0.9,
            0,
            0,
            1.00,
            1.00,
            -1.00,
            -1.00,
            -1.00,
            -1.00,
            -1.00,
            -1.00,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
        },
        std::vector<T_IND>{0, -1},
        std::vector<T_IND>{1, 0},
        "multiclass_nms_post_threshold"};

    const auto indices_size = param.num_batches * param.num_boxes;
    const auto filled_indices = param.expected_selected_indices.size();
    param.expected_selected_indices.resize(indices_size);
    for (auto i = filled_indices; i < indices_size; ++i) {
        param.expected_selected_indices[i] = -1;
    }

    const auto outputs_size = param.num_batches * param.num_classes * param.num_boxes * 6;
    const auto filled_outputs = param.expected_selected_outputs.size();
    param.expected_selected_outputs.resize(outputs_size);
    for (auto i = filled_outputs; i < outputs_size; ++i) {
        param.expected_selected_outputs[i] = -1.0;
    }

    return {param};
}

#define INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(input_type, output_type, func)                                      \
    using multiclass_nms_gpu_test_##input_type##output_type##func = multiclass_nms_test<input_type, output_type>; \
    TEST_P(multiclass_nms_gpu_test_##input_type##output_type##func, test) { ASSERT_NO_FATAL_FAILURE(test()); }    \
    INSTANTIATE_TEST_SUITE_P(                                                                                     \
        multiclass_nms_test_##input_type##output_type##func,                                                      \
        multiclass_nms_gpu_test_##input_type##output_type##func,                                                  \
        testing::Combine(testing::ValuesIn(func<input_type, output_type>()), testing::ValuesIn(layout_formats)),  \
        multiclass_nms_gpu_test_##input_type##output_type##func::PrintToStringParamName);

 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_nms_score_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_nms_by_class_id_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_nms_output_type_i32_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_nms_two_batches_two_classes_by_score_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_nms_two_batches_two_classes_by_class_id_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float,
                                       int32_t,
                                       getMulticlass_nms_two_batches_two_classes_by_score_cross_batch_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_nms_flipped_coordinates_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_nms_identical_boxes_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_limit_output_size_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_single_box_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_IOU_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_IOU_and_scores_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_no_output_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_background_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_keep_top_k_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_eta_input)
INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_keep_post_threshold_inputs)

 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_nms_score_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_nms_by_class_id_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_nms_output_type_i32_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_nms_two_batches_two_classes_by_score_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_nms_two_batches_two_classes_by_class_id_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16,
                                       int32_t,
                                       getMulticlass_nms_two_batches_two_classes_by_score_cross_batch_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_nms_flipped_coordinates_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_nms_identical_boxes_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_limit_output_size_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_single_box_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_IOU_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_IOU_and_scores_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_no_output_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_background_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_keep_top_k_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_eta_input)
INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_keep_post_threshold_inputs)

 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_nms_score_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_nms_by_class_id_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_nms_output_type_i32_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_nms_two_batches_two_classes_by_score_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_nms_two_batches_two_classes_by_class_id_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t,
                                       int32_t,
                                       getMulticlass_nms_two_batches_two_classes_by_score_cross_batch_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_nms_flipped_coordinates_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_nms_identical_boxes_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_limit_output_size_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_single_box_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_IOU_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_IOU_and_scores_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_no_output_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_background_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_keep_top_k_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_eta_input)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_keep_post_threshold_inputs)

 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_nms_three_inputs_blocked_layout)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_nms_three_inputs_blocked_layout)
 INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_nms_three_inputs_blocked_layout)

INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(float, int32_t, getMulticlass_keep_three_inputs)
INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(FLOAT16, int32_t, getMulticlass_keep_three_inputs)
INSTANTIATE_MULTICLASS_NMS_TEST_SUITE(half_t, int32_t, getMulticlass_keep_three_inputs)

#undef INSTANTIATE_MULTICLASS_NMS_TEST_SUITE

};  // namespace