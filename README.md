# Fork Changes

This fork was made to refactor the console application into a light native dll. It refactors the main function into an external function which takes in a struct with possible configurations. This way the ASTC codec can be implemented more fluently into other projects.
It's officially used in Kuriimu2, a flexible fan translation tool for any platform. More specifically the Kanvas image library of this project.

## Changes

Added:
* astc_export.cpp
* astc_export_struct.h

Modified:
* astc_toplevel.cpp

## Exports

* cli* CreateContext();
* void SetMethod(cli *ctx, int method);
* void SetDecodeMode(cli *ctx, int mode);
* void SetInputFile(cli *ctx, char *input);
* void SetOutputFile(cli *ctx, char *input);
* void SetBlockMode(cli *ctx, int blockMode);
* void SetSpeedMode(cli *ctx, int speedMode);
* void SetThreadCount(cli *ctx, int threadCount);
* void DisposeContext(cli *ctx);
* int ConvertImage(cli *ctx);

## How to use

First create a context with "CreateContext". This will create a struct with all possible configurations possible for image creation.
All values of the struct will be set to a default value:
* decodeMode: Decompression
* inputFile: "input.bin"
* outputFile: "output.bin"
* blockMode: ASTC4x4
* SpeedMode: Medium
* ThreadCount: 4

Re-set the configurations you want to change with the "Set"-Exports.

Dispose and remove the context from the heap with "DisposeContext".

Finally use "ConvertImage" to call the main function of the former console application to get your image en-/decoded.

# Credits

Credits go to ARM, the Khronos Group and all contributors to the original repository for making this en-/decoder possible in the first place. Special thanks go to WerWolv98 (https://github.com/WerWolv98) who helped me figuring out most stuff about exports and writing mostly safe and memory-leak free code. Without him these modifications weren't possible (or as safe).

# Rights and license

These modifications doesn't violate any of the terms and conditions in the original license. You may consult license.txt in this fork or the original repository to read anything about the original license and its terms and conditions.
This modification is used in Kuriimu2, which repository is to be found here: https://github.com/FanTranslatorsInternational/Kuriimu2
