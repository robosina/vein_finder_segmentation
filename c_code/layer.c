#include "layer.h"
#include <stdlib.h>

void free_layer(layer l)
{
    if(l.weight)
    {
        free(l.weight);
    }
}
