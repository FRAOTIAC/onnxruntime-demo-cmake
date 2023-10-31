#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>
//#include "cuda_provider_factory.h"
#include <array>
#include <iostream>
#include <iterator>
#include <filesystem>
#include <algorithm>
using namespace std;
// WARNING: to run this example, you need to install cuDNN v7 on your system

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <path_to_model.onnx>\n";
        return -1;
    }
    std::filesystem::path modelPath = argv[1];
    // gives access to the underlying API (you can optionally customize log)
    // you can create one environment per process (each environment manages an internal thread pool)
    Ort::Env env;

    Ort::SessionOptions session_options;
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    Ort::Session session{env,
                         modelPath.c_str(),
                         session_options};

    // cuda should be here
    auto providers = Ort::GetAvailableProviders();

    std::cout << "Available providers: ";
    std::copy(begin(providers), end(providers), std::ostream_iterator<string>(std::cout, " "));
    std::cout << "\n";

    // Ort::Session gives access to input and output information:
    // - count
    // - name
    // - shape and type
    std::cout << "Number of model inputs: " << session.GetInputCount() << "\n";
    std::cout << "Number of model outputs: " << session.GetOutputCount() << "\n";

    // you can customize how allocation works. Let's just use a default allocator provided by the library
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> inputNodeNames; //
    std::vector<Ort::AllocatedStringPtr> inputNodeNameAllocatedStrings; // <-- newly added
    auto inputNodesNum = session.GetInputCount();
    for (int i = 0; i < inputNodesNum; i++) {
        auto input_name = session.GetInputNameAllocated(i, allocator);
        inputNodeNames.push_back(input_name);
        inputNodeNameAllocatedStrings.push_back(Ort::AllocatedStringPtr(input_name, allocator));
        std::cout << "Input " << i << " : " << input_name << "\n";
    }

    // get input and output names
//	auto* inputName = session.GetInputName(0, allocator);
//    auto *inputName = session.GetInputNameAllocated(0, allocator).get();
//    std::cout << "Input name: " << inputName << "\n";

//	auto* outputName = session.GetOutputName(0, allocator);
    auto output = session.GetOutputNameAllocated(0, allocator);
    auto *outputName = session.GetOutputNameAllocated(0, allocator).get();
    std::cout << "Output name: " << outputName << "\n";

    // get input shape
    auto inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    // set some input values
    std::vector<float> inputValues = {4, 5, 6};

    // where to allocate the tensors
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // create the input tensor (this is not a deep copy!)
    auto inputOnnxTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputValues.data(), inputValues.size(),
                                                           inputShape.data(), inputShape.size());
    inputName = "data";
    outputName = "output1";

    std::array<const char *, 1> inputNamesArray = {inputName};
    std::array<const char *, 1> outputNamesArray = {outputName};
    // the API needs the array of inputs you set and the array of outputs you get

    // finally run the inference, using API Run() method above
    auto outputValues = session.Run(
            Ort::RunOptions{nullptr}, // e.g. set a verbosity level only for this run
            inputNamesArray.data(), &inputOnnxTensor, 1, // input
            outputNamesArray.data(), 1); // output

    std::cout << "Output value: " << outputValues[0] << "\n";
    // extract first (and only) output
    auto &output1 = outputValues[0];
    const auto *floats = output1.GetTensorMutableData<float>();
    const auto floatsCount = output1.GetTensorTypeAndShapeInfo().GetElementCount();

    // just print the output values
    std::copy_n(floats, floatsCount, ostream_iterator<float>(std::cout, " "));

    std::cout << "run finished\n";
    // closing boilerplate
    allocator.Free(inputName);
    allocator.Free(outputName);
}
