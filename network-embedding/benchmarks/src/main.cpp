#include "Train.hpp"
#include <string>

int main(int argc,char**argv)
{
	SamplingMethod sMethod = SamplingMethod::UniformDistribution;
    int n = 100;
    double rate = 0.001;
    double margin = 1;
    cout<<"size = "<<n<<endl;
    cout<<"learing rate = "<<rate<<endl;
    cout<<"margin = "<<margin<<endl;

    Train trainer(n, n, rate, margin, sMethod);
	trainer.readData("./initData/realtionEmbedding.data", "./initData/entityEmbedding.data", "./initData/network.data");
    trainer.run();
    trainer.writeData("./finalEmbedding/relation.data", "/finalEmbedding/entity.data" );
}


