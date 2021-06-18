#include <iostream>

#include "GeneralEngine.h"

int main(int argc, char* argv[])
{
    if (argc != 2){
        std::cerr << "You must pass path to .engine file!";
        exit(1);
    }
    GeneralInferenceEngine engine(argv[1]);
}