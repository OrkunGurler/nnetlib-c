#include "activ.h"

typedef struct Nodes
{
    int input, hidden, output;
} Nodes;
Nodes set_nodes(int input, int hidden, int output);

typedef struct Network
{
    Nodes nodes;

    mx weights_ih;
    mx bias_ih;

    mx weights_ho;
    mx bias_ho;

    Activation activation;

    float learning_rate;
} Network;
Network create_network(Nodes node, Activation func, float learning_rate);

mx predict(Network network, mx input);
void train(Network network, mx inputs, mx targets);