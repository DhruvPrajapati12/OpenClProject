#include <stdio.h>
#include <stdlib.h>

static char *load_file(const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);

    char *src = malloc(size + 1);
    fread(src, 1, size, fp);
    src[size] = '\0';

    fclose(fp);
    return src;
}
