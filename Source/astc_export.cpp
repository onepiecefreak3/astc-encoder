#pragma once

#include "astc_export_struct.h"
#include "astc_codec_internals.h"

extern "C" DllExport cli* CreateContext();
extern "C" DllExport void SetMethod(cli *ctx, int method);
extern "C" DllExport void SetDecodeMode(cli *ctx, int mode);
extern "C" DllExport void SetInputFile(cli *ctx, char *input);
extern "C" DllExport void SetOutputFile(cli *ctx, char *input);
extern "C" DllExport void SetBlockMode(cli *ctx, int blockMode);
extern "C" DllExport void SetSpeedMode(cli *ctx, int speedMode);
extern "C" DllExport void SetThreadCount(cli *ctx, int threadCount);
extern "C" DllExport void DisposeContext(cli *ctx);


DllExport cli* CreateContext() {
	cli *input = new cli();

	input->method = Decompression;
	input->decodeMode = DECODE_LDR;
	input->infile = nullptr;
	input->outfile = nullptr;
	input->blockSize = ASTC4x4;
	input->threadCount = 4;

	return input;
}

DllExport void SetMethod(cli *ctx, int method) {
	ctx->method = static_cast<Method>(method);
}

DllExport void SetDecodeMode(cli *ctx, int mode) {
	ctx->decodeMode = static_cast<astc_decode_mode>(mode);
}

DllExport void SetInputFile(cli *ctx, char *input) {
	if (ctx->infile != nullptr)
		delete[] ctx->infile;
	ctx->infile = new char[strlen(input) + 1];
	strcpy(ctx->infile, input);
}

DllExport void SetOutputFile(cli *ctx, char *input) {
	if (ctx->outfile != nullptr)
		delete[] ctx->outfile;
	ctx->outfile = new char[strlen(input) + 1];
	strcpy(ctx->outfile, input);
}

DllExport void SetBlockMode(cli *ctx, int blockSize) {
	ctx->blockSize = static_cast<BlockSize>(blockSize);
}

DllExport void SetSpeedMode(cli *ctx, int speedMode) {
	ctx->speedMode = static_cast<SpeedMode>(speedMode);
}

DllExport void SetThreadCount(cli *ctx, int threadCount) {
	ctx->threadCount = threadCount;
}

DllExport void DisposeContext(cli *ctx) {
	delete ctx;
	ctx = nullptr;
}