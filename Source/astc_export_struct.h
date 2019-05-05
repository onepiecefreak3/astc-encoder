#pragma once

#include "astc_codec_internals.h"

#define DllExport   __declspec( dllexport )  

typedef enum : int {
	Compression = 0,
	Decompression = 1,
	DoBoth = 2,
	Compare = 4
} Method;

typedef enum : int {
	ASTC4x4 = 0,
	ASTC5x4 = 1,
	ASTC5x5 = 2,
	ASTC6x5 = 3,
	ASTC6x6 = 4,
	ASTC8x5 = 5,
	ASTC8x6 = 6,
	ASTC8x8 = 7,
	ASTC10x5 = 8,
	ASTC10x6 = 9,
	ASTC10x8 = 10,
	ASTC10x10 = 11,
	ASTC12x10 = 12,
	ASTC12x12 = 13,

	ASTC3x3x3 = 14,
	ASTC4x3x3 = 15,
	ASTC4x4x3 = 16,
	ASTC4x4x4 = 17,
	ASTC5x4x4 = 18,
	ASTC5x5x4 = 19,
	ASTC5x5x5 = 20,
	ASTC6x5x5 = 21,
	ASTC6x6x5 = 22,
	ASTC6x6x6 = 23,
} BlockSize;

typedef enum : int {
	VeryFast = 0,
	Fast = 1,
	Medium = 2,
	Thorough = 3,
	Exhaustive = 4
} SpeedMode;

typedef struct {
	Method method;
	astc_decode_mode decodeMode;
	char* infile;
	char* outfile;
	BlockSize blockSize;
	SpeedMode speedMode;
	int threadCount;
} cli;