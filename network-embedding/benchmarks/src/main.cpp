#include "Train.hpp"
#include <string>


int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++)  {

    if (!strcmp(str, argv[a])) {

      if (a == argc - 1) {
        printf("Argument missing for %s\n", str);
        exit(1);
      }
      return a;
   }
  return -1;
}

int main(int argc,char**argv)
{
	SamplingMethod sMethod = UniformDistribution;
    int n = 100;
    double rate = 0.001;
    double margin = 1;
    int i;
    if ((i = ArgPos((char *)"-size", argc, argv)) > 0) n = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-method", argc, argv)) > 0) method = atoi(argv[i + 1]);
    cout<<"size = "<<n<<endl;
    cout<<"learing rate = "<<rate<<endl;
    cout<<"margin = "<<margin<<endl;

    Train trainer(n, n, rate, margin, sMethod);

}


