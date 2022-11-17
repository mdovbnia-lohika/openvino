// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/reorder.hpp>

using namespace cldnn;
using namespace ::tests;

template<typename T>
std::vector<T> generate_random_input(size_t b, size_t f, size_t z, size_t y, size_t x, int min, int max) {
    static std::default_random_engine generator(random_seed);
    int k = 8; // 1/k is the resolution of the floating point numbers
    std::uniform_int_distribution<int> distribution(k * min, k * max);
    std::vector<T> v(b*f*x*y*z);
    for (size_t i = 0; i < b*f*x*y*z; ++i) {
        v[i] = (T)distribution(generator);
        v[i] /= k;
    }
    return v;
}

template<typename T>
std::vector<T> generate_random_input(size_t b, size_t f, size_t y, size_t x, int min, int max) {
    return generate_random_input<T>(b, f, 1, y, x, -min, max);
}

TEST(crop_gpu, basic_in2x3x2x2_crop_all) {
    //  Reference  : 1x2x2x2
    //  Input      : 2x3x4x5
    //  Output     : 1x2x2x3

    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 3;
    auto x_size = 4;
    auto y_size = 5;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 2;

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<float> input_vec = generate_random_input<float>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (f + feature_num * (x + x_size * y));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in2x2x2x3_crop_all) {
    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 1;
    auto crop_y_size = y_size - 1;

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<float> input_vec;
    for (int i = 0; i < batch_num * feature_num * y_size * x_size; i++)
        input_vec.push_back(static_cast<float>(i));
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    printf("Results:\n");
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (f + feature_num * (x + x_size * y));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i32_in2x3x2x2_crop_all) {
    //  Reference  : 1x2x2x2
    //  Input      : 2x3x4x5
    //  Output     : 1x2x2x3

    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 3;
    auto x_size = 4;
    auto y_size = 5;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 2;

    auto input = engine.allocate_memory({ data_types::i32, format::yxfb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<int32_t> input_vec = generate_random_input<int32_t>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (f + feature_num * (x + x_size * y));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i64_in2x3x2x2_crop_all) {
    //  Reference  : 1x2x2x2
    //  Input      : 2x3x4x5
    //  Output     : 1x2x2x3

    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 3;
    auto x_size = 4;
    auto y_size = 5;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 2;

    auto input = engine.allocate_memory({ data_types::i64, format::yxfb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<int64_t> input_vec = generate_random_input<int64_t>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (f + feature_num * (x + x_size * y));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in2x3x2x2_crop_all_bfyx) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = engine.allocate_memory({ data_types::f32,format::bfyx,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, {0, 0, 0, 0} ));

    std::vector<float> input_vec = generate_random_input<float>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    std::vector<float> a;
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = x + x_size * (y + y_size * (f + feature_num * b));
                    int output_linear_id = x + crop_x_size * (y + crop_y_size * (f + crop_feature_num * b));
                    a.push_back(output_ptr[output_linear_id]);
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i32_in2x3x2x2_crop_all_bfyx) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = engine.allocate_memory({ data_types::i32,format::bfyx,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<int32_t> input_vec = generate_random_input<int32_t>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    std::vector<int32_t> a;
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = x + x_size * (y + y_size * (f + feature_num * b));
                    int output_linear_id = x + crop_x_size * (y + crop_y_size * (f + crop_feature_num * b));
                    a.push_back(output_ptr[output_linear_id]);
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i64_in2x3x2x2_crop_all_bfyx) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = engine.allocate_memory({ data_types::i64,format::bfyx,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<int64_t> input_vec = generate_random_input<int64_t>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());
    std::vector<int64_t> a;
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = x + x_size * (y + y_size * (f + feature_num * b));
                    int output_linear_id = x + crop_x_size * (y + crop_y_size * (f + crop_feature_num * b));
                    a.push_back(output_ptr[output_linear_id]);
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in2x3x2x2_crop_all_fyxb) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = engine.allocate_memory({ data_types::f32,format::fyxb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, {0, 0, 0, 0} ));

    std::vector<float> input_vec = generate_random_input<float>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (x + x_size * (y + y_size * f));
                    int output_linear_id = b + crop_batch_num * (x + crop_x_size * (y + crop_y_size * f));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i32_in2x3x2x2_crop_all_fyxb) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = engine.allocate_memory({ data_types::i32,format::fyxb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<int32_t> input_vec = generate_random_input<int32_t>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (x + x_size * (y + y_size * f));
                    int output_linear_id = b + crop_batch_num * (x + crop_x_size * (y + crop_y_size * f));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i64_in2x3x2x2_crop_all_fyxb) {
    //  Reference  : 3x1x2x2
    //  Input      : 6x2x4x3
    //  Output     : 3x1x2x2

    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;

    auto input = engine.allocate_memory({ data_types::i64,format::fyxb,{ batch_num, feature_num, x_size, y_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size }, { 0, 0, 0, 0 }));

    std::vector<int64_t> input_vec = generate_random_input<int64_t>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = b + batch_num * (x + x_size * (y + y_size * f));
                    int output_linear_id = b + crop_batch_num * (x + crop_x_size * (y + crop_y_size * f));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in2x3x2x2_crop_offsets) {
    //  Reference  : 1x2x2x1
    //  Offsets    : 1x0x1x1
    //  Input      : 2x2x3x2
    //  Output     : 1x2x2x1

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   0.5 -0.5  -15
    //  f1: b0:  5    6  -12   b1:   1.5  5.2  -13
    //  f1: b0:  7    8  -16   b1:   12   8    -17

    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num;
    auto crop_x_size = x_size - 1;
    auto crop_y_size = y_size - 1;

    auto batch_offset = 1;
    auto feature_offset = 0;
    auto x_offset = 1;
    auto y_offset = 1;

    auto input = engine.allocate_memory({ data_types::f32, format::yxfb, { tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num)), { tensor(feature(0)) }));

    std::vector<float> input_vec = { 1.f, 0.f, 5.f, 1.5f,
        2.f, 0.f, 6.f, 5.2f,
        -10.f, -11.f, -12.f, -13.f,
        3.f, 0.5f, 7.f, 12.f,
        4.f, -0.5f, 8.f, 8.f,
        -14.f, -15.f, -16.f, -17.f };
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = (b + batch_offset) + batch_num * ((f + feature_offset) + feature_num * ((x + x_offset) + x_size * (y + y_offset)));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i32_in2x3x2x2_crop_offsets) {
    //  Reference  : 1x2x2x1
    //  Offsets    : 1x0x1x1
    //  Input      : 2x2x3x2
    //  Output     : 1x2x2x1

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   50   -5   -15
    //  f1: b0:  5    6  -12   b1:   15   52   -13
    //  f1: b0:  7    8  -16   b1:   12   8    -17

    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num;
    auto crop_x_size = x_size - 1;
    auto crop_y_size = y_size - 1;

    auto batch_offset = 1;
    auto feature_offset = 0;
    auto x_offset = 1;
    auto y_offset = 1;

    auto input = engine.allocate_memory({ data_types::i32, format::yxfb,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num)), { tensor(feature(0)) }));

    std::vector<int32_t> input_vec = { 1, 0, 5, 15,
        2, 0, 6, 52,
        -10, -11, -12, -13,
        3, 50, 7, 12,
        4, -5, 8, 8,
        -14, -15, -16, -17 };
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = (b + batch_offset) + batch_num * ((f + feature_offset) + feature_num * ((x + x_offset) + x_size * (y + y_offset)));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_i64_in2x3x2x2_crop_offsets) {
    //  Reference  : 1x2x2x1
    //  Offsets    : 1x0x1x1
    //  Input      : 2x2x3x2
    //  Output     : 1x2x2x1

    //  Input:
    //  f0: b0:  1    2  -10   b1:   0    0    -11
    //  f0: b0:  3    4  -14   b1:   50   -5   -15
    //  f1: b0:  5    6  -12   b1:   15   52   -13
    //  f1: b0:  7    8  -16   b1:   12   8    -17

    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num;
    auto crop_x_size = x_size - 1;
    auto crop_y_size = y_size - 1;

    auto batch_offset = 1;
    auto feature_offset = 0;
    auto x_offset = 1;
    auto y_offset = 1;

    auto input = engine.allocate_memory({ data_types::i64, format::yxfb,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num)), { tensor(feature(0)) }));

    std::vector<int64_t> input_vec = { 1, 0, 5, 15,
        2, 0, 6, 52,
        -10, -11, -12, -13,
        3, 50, 7, 12,
        4, -5, 8, 8,
        -14, -15, -16, -17 };
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = (b + batch_offset) + batch_num * ((f + feature_offset) + feature_num * ((x + x_offset) + x_size * (y + y_offset)));
                    int output_linear_id = b + crop_batch_num * (f + crop_feature_num * (x + crop_x_size * y));
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in1x4x1x1_split) {
    // Tests split with crop implementation
    //                 _CROP_1(1x3x1x1,offset(0x0x0x0))
    //                |
    //  INPUT(1x4x1x1)
    //                |_
    //                  CROP_2(1x1x1x1,offset(0x3x0x0))
    //
    //  Reference1  : 1x3x1x1
    //  Offsets1    : 0x0x0x0
    //  Reference2  : 1x1x1x1
    //  Offsets2    : 0x3x0x0
    //  Input       : 1x4x1x1
    //  Output1     : 1x3x1x1
    //  Output2     : 1x1x1x1

    //  Input:
    //  f0: -1.0
    //  f1:  2.0
    //  f2: -3.0
    //  f3:  4.0

    //  Out1:
    //  f0: -1.0
    //  f1:  2.0
    //  f2: -3.0

    //  Out2:
    //  f0: 4.0
    auto& engine = get_test_engine();

    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 3;
    auto crop_feature_num_2 = 1;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto feature_offset_2 = 3;
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop1", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }));
    topology.add(crop("crop2", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_2)), { tensor(feature(feature_offset_2), spatial(0,0),batch(0)) }));

    std::vector<float> input_vec = { -1.f, 2.f, -3.f, 4.f };
    std::vector<float> out1 = { -1.f, 2.f,-3.f };
    std::vector<float> out2 = { 4.f, };
    set_values(input, input_vec);
    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    bo.set_option(build_option::outputs(topology.get_primitives_ids()));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("crop1").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size();i++)
        EXPECT_EQ(output_ptr[i], out1[i]);

    std::cout << std::endl;
    auto output_2 = outputs.at("crop2").get_memory();
    cldnn::mem_lock<float> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size();i++)
        EXPECT_EQ(output_ptr_2[i], out2[i]);
}

TEST(crop_gpu, basic_in1x4x1x1_crop_pad) {
    auto& engine = get_test_engine();

    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 3;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    padding in_pad({0, 0, 1, 1}, {0, 0, 1, 1});
    auto padded_layout = input->get_layout().with_padding(in_pad);
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("input_reorder", "input", padded_layout));
    topology.add(crop("crop1", "input_reorder", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }));
    topology.add(reorder("out_reorder", "crop1", format::bfyx, data_types::f32));

    std::vector<float> input_vec = { -1.f, 2.f, -3.f, 4.f };
    std::vector<float> out1 = { -1.f, 2.f,-3.f };
    set_values(input, input_vec);
    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("out_reorder").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size();i++)
        EXPECT_EQ(output_ptr[i], out1[i]);
}

TEST(crop_gpu, basic_i32_in1x4x1x1_split) {
    // Tests split with crop implementation
    //                 _CROP_1(1x3x1x1,offset(0x0x0x0))
    //                |
    //  INPUT(1x4x1x1)
    //                |_
    //                  CROP_2(1x1x1x1,offset(0x3x0x0))
    //
    //  Reference1  : 1x3x1x1
    //  Offsets1    : 0x0x0x0
    //  Reference2  : 1x1x1x1
    //  Offsets2    : 0x3x0x0
    //  Input       : 1x4x1x1
    //  Output1     : 1x3x1x1
    //  Output2     : 1x1x1x1

    //  Input:
    //  f0: -1
    //  f1:  2
    //  f2: -3
    //  f3:  4

    //  Out1:
    //  f0: -1
    //  f1:  2
    //  f2: -3

    //  Out2:
    //  f0: 4
    auto& engine = get_test_engine();

    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 3;
    auto crop_feature_num_2 = 1;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto feature_offset_2 = 3;
    auto input = engine.allocate_memory({ data_types::i32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop1", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }));
    topology.add(crop("crop2", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_2)), { tensor(feature(feature_offset_2), spatial(0,0),batch(0)) }));

    std::vector<int32_t> input_vec = { -1, 2, -3, 4 };
    std::vector<int32_t> out1 = { -1, 2,-3 };
    std::vector<int32_t> out2 = { 4, };
    set_values(input, input_vec);
    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    bo.set_option(build_option::outputs(topology.get_primitives_ids()));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("crop1").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size(); i++)
        EXPECT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("crop2").get_memory();
    cldnn::mem_lock<int32_t> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size(); i++)
        EXPECT_EQ(output_ptr_2[i], out2[i]);
}

TEST(crop_gpu, basic_i64_in1x4x1x1_split) {
    // Tests split with crop implementation
    //                 _CROP_1(1x3x1x1,offset(0x0x0x0))
    //                |
    //  INPUT(1x4x1x1)
    //                |_
    //                  CROP_2(1x1x1x1,offset(0x3x0x0))
    //
    //  Reference1  : 1x3x1x1
    //  Offsets1    : 0x0x0x0
    //  Reference2  : 1x1x1x1
    //  Offsets2    : 0x3x0x0
    //  Input       : 1x4x1x1
    //  Output1     : 1x3x1x1
    //  Output2     : 1x1x1x1

    //  Input:
    //  f0: -1
    //  f1:  2
    //  f2: -3
    //  f3:  4

    //  Out1:
    //  f0: -1
    //  f1:  2
    //  f2: -3

    //  Out2:
    //  f0: 4
    auto& engine = get_test_engine();

    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 3;
    auto crop_feature_num_2 = 1;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto feature_offset_2 = 3;
    auto input = engine.allocate_memory({ data_types::i64, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop1", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }));
    topology.add(crop("crop2", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_2)), { tensor(feature(feature_offset_2), spatial(0,0),batch(0)) }));

    std::vector<int64_t> input_vec = { -1, 2, -3, 4 };
    std::vector<int64_t> out1 = { -1, 2,-3 };
    std::vector<int64_t> out2 = { 4, };
    set_values(input, input_vec);
    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    bo.set_option(build_option::outputs(topology.get_primitives_ids()));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("crop1").get_memory();
    cldnn::mem_lock<int64_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size(); i++)
        EXPECT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("crop2").get_memory();
    cldnn::mem_lock<int64_t> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size(); i++)
        EXPECT_EQ(output_ptr_2[i], out2[i]);
}

TEST(crop_gpu, basic_in1x4x1x1_split_w_relu) {
    // Tests split with crop implementation
    //                        _ CROP_1(1x3x1x1,offset(0x0x0x0)) --> RELU
    //                       |
    //  INPUT(1x4x1x1)--RELU
    //                       |_
    //                          CROP_2(1x1x1x1,offset(0x3x0x0)) --> RELU
    //
    //  Reference1  : 1x3x1x1
    //  Offsets1    : 0x0x0x0
    //  Reference2  : 1x1x1x1
    //  Offsets2    : 0x3x0x0
    //  Input       : 1x4x1x1
    //  Output1     : 1x3x1x1
    //  Output2     : 1x1x1x1

    //  Input:
    //  f0: -1.0
    //  f1:  2.0
    //  f2: -3.0
    //  f3:  4.0

    //  Out1:
    //  f0: 0.0
    //  f1: 2.0
    //  f2: 0.0

    //  Out2:
    //  f0: 4.0
    // disable memory pool when we want to check optimized out internal results
    engine_configuration cfg{ false, queue_types::out_of_order, std::string(), priority_mode_types::disabled,  throttle_mode_types::disabled, false /*mem_pool*/ };
    auto engine = engine::create(engine_types::ocl, runtime_types::ocl, cfg);
    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;
    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 3;
    auto crop_feature_num_2 = 1;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto feature_offset_2 = 3;
    auto input = engine->allocate_memory({ data_types::f32, format::bfyx,{ tensor(spatial(x_size, y_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(activation("relu", "input", activation_func::relu));
    topology.add(crop("crop1", "relu", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }));
    topology.add(crop("crop2", "relu", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_2)), { tensor(feature(feature_offset_2), spatial(0,0),batch(0)) }));
    topology.add(activation("relu1", "crop1", activation_func::relu));
    topology.add(activation("relu2", "crop2", activation_func::relu));

    std::vector<float> input_vec = { -1.f, 2.f, -3.f, 4.f };
    std::vector<float> out1 = { 0.f, 2.f,0.f };
    std::vector<float> out2 = { 4.f, };
    set_values(input, input_vec);
    build_options bo;
    bo.set_option(build_option::optimize_data(true));
    bo.set_option(build_option::debug(true)); //required to have optimized crop despite the fact that it's specified as an output

    network network(*engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("relu1").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    // check if crop has been executed in place
    auto in_place = engine->is_the_same_buffer(*outputs.at("crop1").get_memory(), *outputs.at("relu").get_memory());
    EXPECT_TRUE(in_place);

    for (size_t i = 0; i < out1.size();i++)
        EXPECT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("relu2").get_memory();
    cldnn::mem_lock<float> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size();i++)
        EXPECT_EQ(output_ptr_2[i], out2[i]);
}

TEST(crop_gpu, basic_in3x1x2x2x1_crop_all_bfzyx) {
    //  Reference  : 3x1x2x2x1
    //  Input      : 6x2x4x3x2
    //  Output     : 3x1x2x2x1

    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;
    auto z_size = 2;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;
    auto crop_z_size = z_size - 1;

    auto input = engine.allocate_memory({ data_types::f32,format::bfzyx,{ batch_num, feature_num, x_size, y_size, z_size } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", "input", { crop_batch_num, crop_feature_num, crop_x_size, crop_y_size, crop_z_size }, { 0, 0, 0, 0, 0 }));

    std::vector<float> input_vec = generate_random_input<float>(batch_num, feature_num, y_size, x_size, -10, 10);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int z = 0; z < crop_z_size; ++z) { //Z
                for (int y = 0; y < crop_y_size; ++y) { //Y
                    for (int x = 0; x < crop_x_size; ++x) { //X
                        int linear_id = x + x_size * (y + y_size * (z + z_size * (f + feature_num * b)));
                        int output_linear_id = x + crop_x_size * (y + crop_y_size * (z + crop_z_size * (f + crop_feature_num * b)));
                        EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                    }
                }
            }
        }
    }
}

TEST(crop_gpu, basic_in3x1x3x2x2x1_crop_all_bfwzyx) {
    //  Reference  : 3x1x3x2x2x1
    //  Input      : 6x2x6x4x3x2
    //  Output     : 3x1x3x2x2x1

    auto& engine = get_test_engine();

    auto batch_num = 6;
    auto feature_num = 2;
    auto x_size = 4;
    auto y_size = 3;
    auto z_size = 2;
    auto w_size = 6;

    auto crop_batch_num = batch_num - 3;
    auto crop_feature_num = feature_num - 1;
    auto crop_x_size = x_size - 2;
    auto crop_y_size = y_size - 1;
    auto crop_z_size = z_size - 1;
    auto crop_w_size = w_size - 3;

    tensor in_size = tensor(format::bfwzyx, { batch_num, feature_num, w_size, z_size, y_size, x_size });
    tensor crop_size = tensor(format::bfwzyx, { crop_batch_num, crop_feature_num, crop_w_size, crop_z_size, crop_y_size, crop_x_size });
    auto input = engine.allocate_memory({ data_types::f32,format::bfwzyx, in_size });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(crop("crop", "input", crop_size, tensor{ 0 }));

    VVVVVVF<float> input_rnd = generate_random_6d<float>(batch_num, feature_num, w_size, z_size, y_size, x_size, -10, 10);
    VF<float> input_vec = flatten_6d<float>(format::bfwzyx, input_rnd);
    set_values(input, input_vec);

    network network(engine, topology);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int w = 0; w < crop_w_size; ++w) { //W
                for (int z = 0; z < crop_z_size; ++z) { //Z
                    for (int y = 0; y < crop_y_size; ++y) { //Y
                        for (int x = 0; x < crop_x_size; ++x) { //X
                            int linear_id = x + x_size * (y + y_size * (z + z_size * (w + w_size * (f + feature_num * b))));
                            int output_linear_id = x + crop_x_size * (y + crop_y_size * (z + crop_z_size * (w + crop_w_size * (f + crop_feature_num * b))));
                            EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                        }
                    }
                }
            }
        }
    }
}

// batch size, input feature, crop out feature, (in_out format, crop format)
using crop_test_params = std::tuple<size_t, size_t, size_t, std::pair<cldnn::format,cldnn::format>>;

class crop_gpu : public ::testing::TestWithParam<crop_test_params> {};

TEST_P(crop_gpu, pad_test) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto batch_num = std::get<0>(p);
    auto feature_num = std::get<1>(p);
    auto x_size = 1;
    auto y_size = 1;
    auto z_size = 1;

    auto crop_batch_num = batch_num;
    auto crop_feature_num_1 = std::get<2>(p);
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto crop_z_size = 1;
    auto feature_offset_1 = feature_num - crop_feature_num_1;

    auto in_out_format = std::get<3>(p).first;
    auto crop_format = std::get<3>(p).second;

    auto input = engine.allocate_memory({ data_types::f32, in_out_format, { tensor(spatial(x_size, y_size, z_size), feature(feature_num), batch(batch_num)) } });

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(reorder("reorder", "input", crop_format, data_types::f32));
    topology.add(crop("crop1", "reorder", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size, crop_z_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0,0), batch(0)) }));
    topology.add(reorder("out", "crop1", in_out_format, data_types::f32));

    std::vector<float> input_vec;
    std::vector<float> res;
    std::vector<float> input_data;
    std::vector<float> res_data;
    for (size_t i = 0; i < feature_num; i++) {
        input_data.push_back(static_cast<float>(i));
    }
    for (size_t i = 0; i < crop_feature_num_1; i++) {
        res_data.push_back(input_data[feature_offset_1 + i]);
    }
    for (size_t i = 0; i < batch_num; i++) {
        input_vec.insert(input_vec.end(), input_data.begin(), input_data.end());
        res.insert(res.end(), res_data.begin(), res_data.end());
    }
    set_values(input, input_vec);
    build_options bo;
    bo.set_option(build_option::optimize_data(true));

    network network(engine, topology, bo);
    network.set_input_data("input", input);
    auto outputs = network.execute();

    auto output = outputs.at("out").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < res.size(); i++)
        EXPECT_EQ(output_ptr[i], res[i]);
}

static std::vector<std::pair<cldnn::format,cldnn::format>> formats = {
    std::make_pair<cldnn::format, cldnn::format>(format::bfyx, format::b_fs_yx_fsv16),
    std::make_pair<cldnn::format, cldnn::format>(format::bfzyx, format::b_fs_zyx_fsv16),
    std::make_pair<cldnn::format, cldnn::format>(format::bfyx, format::bs_fs_yx_bsv16_fsv16),
    std::make_pair<cldnn::format, cldnn::format>(format::bfzyx, format::bs_fs_zyx_bsv16_fsv16),
    };
static std::vector<size_t> batches = {1, 8, 16, 17};
static std::vector<size_t> in_features = {18, 24, 32};
static std::vector<size_t> crop_features = {4, 8, 12, 17};

INSTANTIATE_TEST_SUITE_P(crop_test, crop_gpu,
                        ::testing::Combine(
                                ::testing::ValuesIn(batches),
                                ::testing::ValuesIn(in_features),
                                ::testing::ValuesIn(crop_features),
                                ::testing::ValuesIn(formats)
                                ));


TEST(crop_gpu, dynamic_i32_in2x3x2x2_crop_offsets) {
    auto& engine = get_test_engine();

    auto batch_num = 2;
    auto feature_num = 2;
    auto x_size = 3;
    auto y_size = 2;

    auto crop_batch_num = batch_num - 1;
    auto crop_feature_num = feature_num;
    auto crop_x_size = x_size - 1;
    auto crop_y_size = y_size - 1;

    auto batch_offset = 1;
    auto feature_offset = 0;
    auto x_offset = 1;
    auto y_offset = 1;

    auto input_dyn_layout    = layout{ ov::PartialShape{ov::Dimension(1, 10), feature_num, y_size, x_size}, data_types::f32, format::bfyx };
    auto input_actual_layout = layout{ ov::PartialShape{batch_num, feature_num, y_size, x_size}, data_types::f32, format::bfyx };

    auto input = engine.allocate_memory(input_actual_layout);

    topology topology;
    topology.add(input_layout("input", input_dyn_layout));
    topology.add(crop("crop", "input", tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num)), { tensor(feature(0)) }));

    std::vector<int32_t> input_vec = { 1, 0, 5, 15,
        2, 0, 6, 52,
        -10, -11, -12, -13,
        3, 50, 7, 12,
        4, -5, 8, 8,
        -14, -15, -16, -17 };
    set_values(input, input_vec);
    build_options bo;
    bo.set_option(build_option::allow_new_shape_infer(true));
    network network(engine, topology, bo);

    network.set_input_data("input", input);

    auto outputs = network.execute();

    auto output = outputs.at("crop").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    for (int b = 0; b < crop_batch_num; ++b) { //B
        for (int f = 0; f < crop_feature_num; ++f) { //F
            for (int y = 0; y < crop_y_size; ++y) { //Y
                for (int x = 0; x < crop_x_size; ++x) { //X
                    int linear_id = (b + batch_offset) * (feature_num * y_size * x_size) + (f + feature_offset) * (y_size * x_size) + (y + y_offset) * x_size + (x + x_offset);
                    int output_linear_id = b * (crop_feature_num * crop_y_size * crop_x_size) + f * (crop_y_size * crop_x_size) + y * crop_x_size + x;
                    EXPECT_EQ(output_ptr[output_linear_id], input_vec[linear_id]);
                }
            }
        }
    }
}

TEST(crop_gpu, dynamic_in1x4x1x1_split) {
    auto& engine = get_test_engine();

    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 2;
    auto crop_feature_num_2 = 2;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto feature_offset_2 = 2;

    auto input_dyn_layout    = layout{ ov::PartialShape{ov::Dimension(1, 10), feature_num, y_size, x_size}, data_types::f32, format::bfyx };
    auto input_actual_layout = layout{ ov::PartialShape{batch_num, feature_num, y_size, x_size}, data_types::f32, format::bfyx };

    auto input_mem = engine.allocate_memory(input_actual_layout);
    auto data_mem = engine.allocate_memory({ {}, data_types::i64, format::bfyx });
    set_values(data_mem, {1});

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::split;
    size_t num_splits = 2;
    topology topology;
    topology.add(input_layout("input", input_dyn_layout));
    topology.add(data("data", data_mem));
    topology.add(crop("crop1", {"input", "data"}, tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }, op_mode, 0, num_splits));
    topology.add(crop("crop2", {"input", "data"}, tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_2)), { tensor(feature(feature_offset_2), spatial(0,0),batch(0)) }, op_mode, 1, num_splits));

    std::vector<int32_t> input_vec = { -1, 2, -3, 4 };
    std::vector<int32_t> out1 = { -1, 2 };
    std::vector<int32_t> out2 = { -3, 4 };
    set_values(input_mem, input_vec);
    build_options bo;
    bo.set_option(build_option::allow_new_shape_infer(true));
    bo.set_option(build_option::optimize_data(true));
    bo.set_option(build_option::outputs(topology.get_primitives_ids()));

    network network(engine, topology, bo);
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();

    auto output = outputs.at("crop1").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size(); i++)
        EXPECT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("crop2").get_memory();
    cldnn::mem_lock<int32_t> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size(); i++)
        EXPECT_EQ(output_ptr_2[i], out2[i]);
}

TEST(crop_gpu, dynamic_in1x4x1x1_varaidic_split) {
    auto& engine = get_test_engine();

    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;

    auto crop_batch_num = 1;
    auto crop_feature_num_1 = 3;
    auto crop_feature_num_2 = 1;
    auto crop_x_size = 1;
    auto crop_y_size = 1;
    auto feature_offset_1 = 0;
    auto feature_offset_2 = 3;

    auto input_dyn_layout    = layout{ ov::PartialShape{ov::Dimension(1, 10), feature_num, y_size, x_size}, data_types::f32, format::bfyx };
    auto input_actual_layout = layout{ ov::PartialShape{batch_num, feature_num, y_size, x_size}, data_types::f32, format::bfyx };

    auto input_mem = engine.allocate_memory(input_actual_layout);
    auto axis_mem = engine.allocate_memory({ {}, data_types::i64, format::bfyx });
    auto splits_length_mem = engine.allocate_memory({ {2}, data_types::i64, format::bfyx });

    cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
    topology topology;
    topology.add(input_layout("input", input_dyn_layout));
    topology.add(data("axis", axis_mem));
    topology.add(data("splits_length", splits_length_mem));
    topology.add(crop("crop1", {"input", "axis", "splits_length"}, tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }, op_mode, 0));
    topology.add(crop("crop2", {"input", "axis", "splits_length"}, tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_2)), { tensor(feature(feature_offset_2), spatial(0,0),batch(0)) }, op_mode, 1));

    std::vector<int32_t> input_vec = { -1, 2, -3, 4 };
    std::vector<int32_t> out1 = { -1, 2, -3 };
    std::vector<int32_t> out2 = { 4 };
    std::vector<int64_t> splits_vec = {3, 1};

    set_values(input_mem, input_vec);
    set_values(axis_mem, {1});
    set_values(splits_length_mem, splits_vec);

    build_options bo;
    bo.set_option(build_option::allow_new_shape_infer(true));
    bo.set_option(build_option::optimize_data(true));
    bo.set_option(build_option::outputs(topology.get_primitives_ids()));

    network network(engine, topology, bo);
    network.set_input_data("input", input_mem);
    auto outputs = network.execute();

    auto output = outputs.at("crop1").get_memory();
    cldnn::mem_lock<int32_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < out1.size(); i++)
        EXPECT_EQ(output_ptr[i], out1[i]);

    auto output_2 = outputs.at("crop2").get_memory();
    cldnn::mem_lock<int32_t> output_ptr_2(output_2, get_test_stream());

    for (size_t i = 0; i < out2.size(); i++)
        EXPECT_EQ(output_ptr_2[i], out2[i]);
}


struct crop_random_test_params {
    data_types data_type;
    int batch_num;
    int feature_num;
    int x_size;
    int y_size;
    tensor input;
    format::type in_format;
    tensor output;
    format::type output_format;
};

struct crop_random_test : testing::TestWithParam<crop_random_test_params> {

    static double get_exectime(const std::map<cldnn::primitive_id, cldnn::network_output>& outputs,
                               const std::string& primitive_id) {
        using namespace std::chrono;
        std::shared_ptr<event> e = outputs.at(primitive_id).get_event();
        e->wait();  // should ensure execution completion, if not segfault will occur
        double avg_time = 0.0;
        auto intervals = e->get_profiling_info();
        for (const auto& q : intervals) {
            if (q.stage != instrumentation::profiling_stage::executing) {
                continue;
            }
            avg_time = duration_cast<duration<double, microseconds::period>>(q.value->value()).count();
            break;
        }
        return avg_time;
    }

    static void print_all_perf(std::map<primitive_id, network_output>& outputs) {
        std::cout << "Print last run time" << std::endl;
        using namespace std::chrono;
        for (const auto& n : outputs) {
            std::shared_ptr<event> e = n.second.get_event();
            auto intervals = e->get_profiling_info();
            double time = 0.0;
            for (const auto& q : intervals) {
                if (q.stage != instrumentation::profiling_stage::executing) {
                    continue;
                }
                time = duration_cast<duration<double, microseconds::period>>(q.value->value()).count();
                break;
            }
            std::cout << n.first << ":" << time << std::endl;
        }
        std::cout << std::endl;
    }

    cldnn::engine_configuration get_profiling_config() {
        // const bool enable_profiling = true;
        std::string sources_dumps_dir = "";
        cldnn::queue_types queue_type = cldnn::queue_types::out_of_order;
        priority_mode_types priority_mode = priority_mode_types::disabled;
        throttle_mode_types throttle_mode = throttle_mode_types::disabled;
        bool use_memory_pool = true;
        bool use_unified_shared_memory = true;
        return engine_configuration(true,
                                    queue_type,
                                    sources_dumps_dir,
                                    priority_mode,
                                    throttle_mode,
                                    use_memory_pool,
                                    use_unified_shared_memory);
    }

    template <typename T>
    void fill_random_typed(memory::ptr mem, int min, int max, int k) {
        auto l = mem->get_layout();
        size_t b = l.batch();
        size_t f = l.feature();
        size_t x = l.spatial(0);
        size_t y = l.spatial(1);
    }

    void fill_random(memory::ptr mem) {
        auto dt = mem->get_layout().data_type;
        switch (dt) {
        case data_types::f32:
            fill_random_typed<float>(mem, -127, 127, 2);
            break;
        case data_types::f16:
            fill_random_typed<FLOAT16>(mem, -127, 127, 2);
            break;
        case data_types::i8:
            fill_random_typed<int8_t>(mem, -127, 127, 1);
            break;
        case data_types::u8:
            fill_random_typed<uint8_t>(mem, 0, 255, 1);
            break;
        default:
            break;
        }
    }

    void execute_perf_test(const crop_random_test_params& params,
                           const std::string& kernel,
                           const bool do_plain = false) {

        auto crop_batch_num = 32;
        auto crop_feature_num_1 = 32;
        auto crop_x_size = 256;
        auto crop_y_size = 256;
        auto feature_offset_1 = 0;

        auto& engine = get_test_engine(get_profiling_config());

        const format fmt_origin = format::bfyx;
        const format working_format = do_plain ? fmt_origin : format(params.in_format);

        auto in_layout = layout(params.data_type, fmt_origin, params.input);
        auto input = engine.allocate_memory(in_layout);
        auto axis_mem = engine.allocate_memory({ {}, data_types::i64, fmt_origin});
        auto splits_length_mem = engine.allocate_memory({ {2}, data_types::i64, fmt_origin });

        fill_random(input);

        cldnn::crop_ngraph_op_mode op_mode = cldnn::crop_ngraph_op_mode::variadic_split;
        topology topology;

        topology.add(input_layout("input", input->get_layout()));
        topology.add(reorder("reorder_input", "input", working_format, params.data_type));
        topology.add(data("axis", axis_mem));
        topology.add(data("splits_length", splits_length_mem));
        topology.add(crop("crop", {"reorder_input", "axis", "splits_length"}, tensor(batch(crop_batch_num), spatial(crop_x_size, crop_y_size), feature(crop_feature_num_1)), { tensor(feature(feature_offset_1), spatial(0,0),batch(0)) }, op_mode, 0));
        topology.add(reorder("crop_out", "crop", fmt_origin, params.data_type));

        set_values(axis_mem, {1});
        std::vector<int64_t> splits_vec = {-1, 2};
        set_values(splits_length_mem, splits_vec);

        auto build_opts_opt = build_options();
        build_opts_opt.set_option(build_option::outputs({"crop_out"}));
        build_opts_opt.set_option(build_option::force_implementations({{{"crop"}, {working_format, kernel}}}));
        build_opts_opt.set_option(build_option::debug(true));

        network network(engine, topology, build_opts_opt);
        network.set_input_data("input", input);
        network.get_program();

        std::map<primitive_id, network_output> result_opt;
        auto r = 100;
        double exectime = 0.f;
        for (int i = 0; i < r; ++i) {
            result_opt = network.execute();
            exectime += get_exectime(result_opt, {"crop"});
        }

        exectime /= r;
        std::string frm_str = format(working_format).to_string();
        std::string input_type = data_type_traits::name(params.data_type);
        std::string is_opt = do_plain ? " not optimazed " : " optimized ";


        std::cout << "Executed time " << " "<< exectime << " " << is_opt << " " << kernel << " "
                  << " input_first(" << params.input.to_string() << ")"
                  << " input_second(" << params.output.to_string() << ")" << frm_str << " " << input_type
                  << " " << exectime << std::endl;
    }
};

TEST_P(crop_random_test, random) {
    auto param = GetParam();
    execute_perf_test(param, "generic_eltwise_ref");
    execute_perf_test(param, "generic_eltwise_ref", true);
}

INSTANTIATE_TEST_SUITE_P(
    crop_random_test_fsv32,
    crop_random_test,
    testing::Values(
        crop_random_test_params{data_types::f16, 1, 4, 1, 1, {32, 64, 256, 256}, format::bs_fs_yx_bsv32_fsv16, {32, 32, 256, 256},
                                format::bs_fs_yx_bsv32_fsv16}
        ));