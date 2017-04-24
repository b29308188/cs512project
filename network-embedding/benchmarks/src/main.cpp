#include "Train.hpp"
#include <unistd.h>
#include <sys/stat.h>
#include <string>

inline bool checkFileExistence(const std::string& fileName);
void checkAllFiles(std::vector<std::string> arr);

int main(int argc,char**argv)
{
	std::string inputDir = argv[1];
	std::string outputDir = argv[2];
	
	std::string inputRelationFile = inputDir + "/relationEmbedding.data";
	std::string inputEntityFile = inputDir + "/entityEmbedding.data";
	std::string networkFile = inputDir + "/network.data";
	
	std::string outputRelationFile = outputDir + "/relation.data";
	std::string outputEntityFile = outputDir + "/entity.data";

	checkAllFiles( {inputRelationFile, inputEntityFile, networkFile, outputRelationFile, outputEntityFile } );

	SamplingMethod sMethod = SamplingMethod::UniformDistribution;
    int n = 300;
    double rate = 0.001;
    double margin = 1;
    cout<<"size = "<<n<<endl;
    cout<<"learing rate = "<<rate<<endl;
    cout<<"margin = "<<margin<<endl;

    Train trainer(n, n, rate, margin, sMethod);
	trainer.readData(inputRelationFile, inputEntityFile, networkFile);
    trainer.run();
	trainer.writeData(outputRelationFile, outputEntityFile);
	return 0;
}


inline bool checkFileExistence(const std::string& fileName) {
	struct stat buffer;
	return (stat(fileName.c_str(), &buffer) == 0);
}
void checkAllFiles(std::vector<std::string> arr){
	for (auto fileName:arr) {
		if (!checkFileExistence(fileName)) {
			std::cerr << fileName << " doesnt exist!" << std::endl;
			exit(2);
		}
	}
}
