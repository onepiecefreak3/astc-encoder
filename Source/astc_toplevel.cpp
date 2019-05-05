/*----------------------------------------------------------------------------*/
/**
 *	This confidential and proprietary software may be used only as
 *	authorised by a licensing agreement from ARM Limited
 *	(C) COPYRIGHT 2011-2013 ARM Limited
 *	ALL RIGHTS RESERVED
 *
 *	The entire notice above must be reproduced on all authorised
 *	copies and copies may only be made to the extent permitted
 *	by a licensing agreement from ARM Limited.
 *
 *	@brief	Top level functions - parsing command line, managing conversions,
 *			etc.
 *
 *			This is also where main() lives.
 */
/*----------------------------------------------------------------------------*/

#include "astc_codec_internals.h"
#include "astc_export_struct.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef WIN32
	#include <sys/time.h>
	#include <pthread.h>
	#include <unistd.h>

	double get_time()
	{
		timeval tv;
		gettimeofday(&tv, 0);

		return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
	}


	int astc_codec_unlink(const char *filename)
	{
		return unlink(filename);
	}

#else
	// Windows.h defines IGNORE, so we must #undef our own version.
	#undef IGNORE

	// Define pthread-like functions in terms of Windows threading API
	#define WIN32_LEAN_AND_MEAN
	#include <windows.h>

	typedef HANDLE pthread_t;
	typedef int pthread_attr_t;

	int pthread_create(pthread_t * thread, const pthread_attr_t * attribs, void *(*threadfunc) (void *), void *thread_arg)
	{
		*thread = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) threadfunc, thread_arg, 0, NULL);
		return 0;
	}

	int pthread_join(pthread_t thread, void **value)
	{
		WaitForSingleObject(thread, INFINITE);
		return 0;
	}

	double get_time()
	{
		FILETIME tv;
		GetSystemTimeAsFileTime(&tv);

		unsigned __int64 ticks = tv.dwHighDateTime;
		ticks = (ticks << 32) | tv.dwLowDateTime;

		return ((double)ticks) / 1.0e7;
	}

	// Define an unlink() function in terms of the Win32 DeleteFile function.
	int astc_codec_unlink(const char *filename)
	{
		BOOL res = DeleteFileA(filename);
		return (res ? 0 : -1);
	}
#endif

#ifdef DEBUG_CAPTURE_NAN
	#ifndef _GNU_SOURCE
		#define _GNU_SOURCE
	#endif

	#include <fenv.h>
#endif

// Define this to be 1 to allow "illegal" block sizes
#define DEBUG_ALLOW_ILLEGAL_BLOCK_SIZES 0

extern int block_mode_histogram[2048];

#ifdef DEBUG_PRINT_DIAGNOSTICS
	int print_diagnostics = 0;
	int diagnostics_tile = -1;
#endif

int print_tile_errors = 0;

int print_statistics = 0;

int progress_counter_divider = 1;

int rgb_force_use_of_hdr = 0;
int alpha_force_use_of_hdr = 0;


static double start_time;
static double end_time;
static double start_coding_time;
static double end_coding_time;


// code to discover the number of logical CPUs available.

#if defined(__APPLE__)
	#define _DARWIN_C_SOURCE
	#include <sys/types.h>
	#include <sys/sysctl.h>
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
	#include <windows.h>
#else
	#include <unistd.h>
#endif



unsigned get_number_of_cpus(void)
{
	unsigned n_cpus = 1;

	#ifdef __linux__
		cpu_set_t mask;
		CPU_ZERO(&mask);
		sched_getaffinity(getpid(), sizeof(mask), &mask);
		n_cpus = 0;
		for (unsigned i = 0; i < CPU_SETSIZE; ++i)
		{
			if (CPU_ISSET(i, &mask))
				n_cpus++;
		}
		if (n_cpus == 0)
			n_cpus = 1;

	#elif defined (_WIN32) || defined(__CYGWIN__)
		SYSTEM_INFO sysinfo;
		GetSystemInfo(&sysinfo);
		n_cpus = sysinfo.dwNumberOfProcessors;

	#elif defined(__APPLE__)
		int mib[4];
		size_t length = 100;
		mib[0] = CTL_HW;
		mib[1] = HW_AVAILCPU;
		sysctl(mib, 2, &n_cpus, &length, NULL, 0);
	#endif

	return n_cpus;
}

void astc_codec_internal_error(const char *filename, int linenum)
{
	printf("Internal error: File=%s Line=%d\n", filename, linenum);
	exit(1);
}

#define MAGIC_FILE_CONSTANT 0x5CA1AB13

struct astc_header
{
	uint8_t magic[4];
	uint8_t blockdim_x;
	uint8_t blockdim_y;
	uint8_t blockdim_z;
	uint8_t xsize[3];			// x-size = xsize[0] + xsize[1] + xsize[2]
	uint8_t ysize[3];			// x-size, y-size and z-size are given in texels;
	uint8_t zsize[3];			// block count is inferred
};


int suppress_progress_counter = 0;
int perform_srgb_transform = 0;

astc_codec_image *load_astc_file(const char *filename, int bitness, astc_decode_mode decode_mode, swizzlepattern swz_decode)
{
	int x, y, z;
	FILE *f = fopen(filename, "rb");
	if (!f)
	{
		printf("Failed to open file %s\n", filename);
		exit(1);
	}
	astc_header hdr;
	size_t hdr_bytes_read = fread(&hdr, 1, sizeof(astc_header), f);
	if (hdr_bytes_read != sizeof(astc_header))
	{
		fclose(f);
		printf("Failed to read file %s\n", filename);
		exit(1);
	}

	uint32_t magicval = hdr.magic[0] + 256 * (uint32_t) (hdr.magic[1]) + 65536 * (uint32_t) (hdr.magic[2]) + 16777216 * (uint32_t) (hdr.magic[3]);

	if (magicval != MAGIC_FILE_CONSTANT)
	{
		fclose(f);
		printf("File %s not recognized\n", filename);
		exit(1);
	}

	int xdim = hdr.blockdim_x;
	int ydim = hdr.blockdim_y;
	int zdim = hdr.blockdim_z;

	if ( (xdim < 3 || xdim > 6 || ydim < 3 || ydim > 6 || zdim < 3 || zdim > 6) &&
	     (xdim < 4 || xdim == 7 || xdim == 9 || xdim == 11 || xdim > 12 ||
	      ydim < 4 || ydim == 7 || ydim == 9 || ydim == 11 || ydim > 12 || zdim != 1) )
	{
		fclose(f);
		printf("File %s not recognized %d %d %d\n", filename, xdim, ydim, zdim);
		exit(1);
	}


	int xsize = hdr.xsize[0] + 256 * hdr.xsize[1] + 65536 * hdr.xsize[2];
	int ysize = hdr.ysize[0] + 256 * hdr.ysize[1] + 65536 * hdr.ysize[2];
	int zsize = hdr.zsize[0] + 256 * hdr.zsize[1] + 65536 * hdr.zsize[2];

	if (xsize == 0 || ysize == 0 || zsize == 0)
	{
		fclose(f);
		printf("File %s has zero dimension %d %d %d\n", filename, xsize, ysize, zsize);
		exit(1);
	}


	int xblocks = (xsize + xdim - 1) / xdim;
	int yblocks = (ysize + ydim - 1) / ydim;
	int zblocks = (zsize + zdim - 1) / zdim;

	uint8_t *buffer = (uint8_t *) malloc(xblocks * yblocks * zblocks * 16);
	if (!buffer)
	{
		fclose(f);
		printf("Ran out of memory\n");
		exit(1);
	}
	size_t bytes_to_read = xblocks * yblocks * zblocks * 16;
	size_t bytes_read = fread(buffer, 1, bytes_to_read, f);
	fclose(f);
	if (bytes_read != bytes_to_read)
	{
		printf("Failed to read file %s\n", filename);
		exit(1);
	}


	astc_codec_image *img = allocate_image(bitness, xsize, ysize, zsize, 0);
	initialize_image(img);

	imageblock pb;
	for (z = 0; z < zblocks; z++)
		for (y = 0; y < yblocks; y++)
			for (x = 0; x < xblocks; x++)
			{
				int offset = (((z * yblocks + y) * xblocks) + x) * 16;
				uint8_t *bp = buffer + offset;
				physical_compressed_block pcb = *(physical_compressed_block *) bp;
				symbolic_compressed_block scb;
				physical_to_symbolic(xdim, ydim, zdim, pcb, &scb);
				decompress_symbolic_block(decode_mode, xdim, ydim, zdim, x * xdim, y * ydim, z * zdim, &scb, &pb);
				write_imageblock(img, &pb, xdim, ydim, zdim, x * xdim, y * ydim, z * zdim, swz_decode);
			}

	free(buffer);

	return img;
}



struct encode_astc_image_info
{
	int xdim;
	int ydim;
	int zdim;
	const error_weighting_params *ewp;
	uint8_t *buffer;
	int *counters;
	int pack_and_unpack;
	int thread_id;
	int threadcount;
	astc_decode_mode decode_mode;
	swizzlepattern swz_encode;
	swizzlepattern swz_decode;
	int *threads_completed;
	const astc_codec_image *input_image;
	astc_codec_image *output_image;
};



void *encode_astc_image_threadfunc(void *vblk)
{
	const encode_astc_image_info *blk = (const encode_astc_image_info *)vblk;
	int xdim = blk->xdim;
	int ydim = blk->ydim;
	int zdim = blk->zdim;
	uint8_t *buffer = blk->buffer;
	const error_weighting_params *ewp = blk->ewp;
	int thread_id = blk->thread_id;
	int threadcount = blk->threadcount;
	int *counters = blk->counters;
	int pack_and_unpack = blk->pack_and_unpack;
	astc_decode_mode decode_mode = blk->decode_mode;
	swizzlepattern swz_encode = blk->swz_encode;
	swizzlepattern swz_decode = blk->swz_decode;
	int *threads_completed = blk->threads_completed;
	const astc_codec_image *input_image = blk->input_image;
	astc_codec_image *output_image = blk->output_image;

	imageblock pb;
	int ctr = thread_id;
	int pctr = 0;

	int x, y, z, i;
	int xsize = input_image->xsize;
	int ysize = input_image->ysize;
	int zsize = input_image->zsize;
	int xblocks = (xsize + xdim - 1) / xdim;
	int yblocks = (ysize + ydim - 1) / ydim;
	int zblocks = (zsize + zdim - 1) / zdim;

	int owns_progress_counter = 0;

	//allocate memory for temporary buffers
	compress_symbolic_block_buffers temp_buffers;
	temp_buffers.ewb = new error_weight_block;
	temp_buffers.ewbo = new error_weight_block_orig;
	temp_buffers.tempblocks = new symbolic_compressed_block[4];
	temp_buffers.temp = new imageblock;
	temp_buffers.planes2 = new compress_fixed_partition_buffers;
	temp_buffers.planes2->ei1 = new endpoints_and_weights;
	temp_buffers.planes2->ei2 = new endpoints_and_weights;
	temp_buffers.planes2->eix1 = new endpoints_and_weights[MAX_DECIMATION_MODES];
	temp_buffers.planes2->eix2 = new endpoints_and_weights[MAX_DECIMATION_MODES];
	temp_buffers.planes2->decimated_quantized_weights = new float[2 * MAX_DECIMATION_MODES * MAX_WEIGHTS_PER_BLOCK];
	temp_buffers.planes2->decimated_weights = new float[2 * MAX_DECIMATION_MODES * MAX_WEIGHTS_PER_BLOCK];
	temp_buffers.planes2->flt_quantized_decimated_quantized_weights = new float[2 * MAX_WEIGHT_MODES * MAX_WEIGHTS_PER_BLOCK];
	temp_buffers.planes2->u8_quantized_decimated_quantized_weights = new uint8_t[2 * MAX_WEIGHT_MODES * MAX_WEIGHTS_PER_BLOCK];
	temp_buffers.plane1 = temp_buffers.planes2;

	for (z = 0; z < zblocks; z++)
		for (y = 0; y < yblocks; y++)
			for (x = 0; x < xblocks; x++)
			{
				if (ctr == 0)
				{
					int offset = ((z * yblocks + y) * xblocks + x) * 16;
					uint8_t *bp = buffer + offset;
				#ifdef DEBUG_PRINT_DIAGNOSTICS
					if (diagnostics_tile < 0 || diagnostics_tile == pctr)
					{
						print_diagnostics = (diagnostics_tile == pctr) ? 1 : 0;
				#endif
						fetch_imageblock(input_image, &pb, xdim, ydim, zdim, x * xdim, y * ydim, z * zdim, swz_encode);
						symbolic_compressed_block scb;
						compress_symbolic_block(input_image, decode_mode, xdim, ydim, zdim, ewp, &pb, &scb, &temp_buffers);
						if (pack_and_unpack)
						{
							decompress_symbolic_block(decode_mode, xdim, ydim, zdim, x * xdim, y * ydim, z * zdim, &scb, &pb);
							write_imageblock(output_image, &pb, xdim, ydim, zdim, x * xdim, y * ydim, z * zdim, swz_decode);
						}
						else
						{
							physical_compressed_block pcb;
							pcb = symbolic_to_physical(xdim, ydim, zdim, &scb);
							*(physical_compressed_block *) bp = pcb;
						}
				#ifdef DEBUG_PRINT_DIAGNOSTICS
					}
				#endif

					counters[thread_id]++;
					ctr = threadcount - 1;

					pctr++;

					// routine to print the progress counter.
					if (suppress_progress_counter == 0 && (pctr % progress_counter_divider) == 0 && print_tile_errors == 0 && print_statistics == 0)
					{
						int do_print = 1;
						// the current thread has the responsibility for printing the progress counter
						// if every previous thread has completed. Also, if we have ever received the
						// responsibility to print the progress counter, we are going to keep it
						// until the thread is completed.
						if (!owns_progress_counter)
						{
							for (i = thread_id - 1; i >= 0; i--)
							{
								if (threads_completed[i] == 0)
								{
									do_print = 0;
									break;
								}
							}
						}
						if (do_print)
						{
							owns_progress_counter = 1;
							int summa = 0;
							for (i = 0; i < threadcount; i++)
								summa += counters[i];
							printf("\r%d", summa);
							fflush(stdout);
						}
					}
				}
				else
					ctr--;
			}

	delete[] temp_buffers.planes2->decimated_quantized_weights;
	delete[] temp_buffers.planes2->decimated_weights;
	delete[] temp_buffers.planes2->flt_quantized_decimated_quantized_weights;
	delete[] temp_buffers.planes2->u8_quantized_decimated_quantized_weights;
	delete[] temp_buffers.planes2->eix1;
	delete[] temp_buffers.planes2->eix2;
	delete   temp_buffers.planes2->ei1;
	delete   temp_buffers.planes2->ei2;
	delete   temp_buffers.planes2;
	delete[] temp_buffers.tempblocks;
	delete   temp_buffers.temp;
	delete   temp_buffers.ewbo;
	delete   temp_buffers.ewb;

	threads_completed[thread_id] = 1;
	return NULL;
}


void encode_astc_image(const astc_codec_image * input_image,
					   astc_codec_image * output_image,
					   int xdim,
					   int ydim,
					   int zdim,
					   const error_weighting_params * ewp, astc_decode_mode decode_mode, swizzlepattern swz_encode, swizzlepattern swz_decode, uint8_t * buffer, int pack_and_unpack, int threadcount)
{
	int i;
	int *counters = new int[threadcount];
	int *threads_completed = new int[threadcount];

	// before entering into the multi-threaded routine, ensure that the block size descriptors
	// and the partition table descriptors needed actually exist.
	get_block_size_descriptor(xdim, ydim, zdim);
	get_partition_table(xdim, ydim, zdim, 0);

	encode_astc_image_info *ai = new encode_astc_image_info[threadcount];
	for (i = 0; i < threadcount; i++)
	{
		ai[i].xdim = xdim;
		ai[i].ydim = ydim;
		ai[i].zdim = zdim;
		ai[i].buffer = buffer;
		ai[i].ewp = ewp;
		ai[i].counters = counters;
		ai[i].pack_and_unpack = pack_and_unpack;
		ai[i].thread_id = i;
		ai[i].threadcount = threadcount;
		ai[i].decode_mode = decode_mode;
		ai[i].swz_encode = swz_encode;
		ai[i].swz_decode = swz_decode;
		ai[i].threads_completed = threads_completed;
		ai[i].input_image = input_image;
		ai[i].output_image = output_image;
		counters[i] = 0;
		threads_completed[i] = 0;
	}

	if (threadcount == 1)
		encode_astc_image_threadfunc(&ai[0]);
	else
	{
		pthread_t *threads = new pthread_t[threadcount];
		for (i = 0; i < threadcount; i++)
			pthread_create(&(threads[i]), NULL, encode_astc_image_threadfunc, (void *)(&(ai[i])));

		for (i = 0; i < threadcount; i++)
			pthread_join(threads[i], NULL);
		delete[]threads;
	}

	delete[]ai;
	delete[]counters;
	delete[]threads_completed;
}


void store_astc_file(const astc_codec_image * input_image,
					 const char *filename, int xdim, int ydim, int zdim, const error_weighting_params * ewp, astc_decode_mode decode_mode, swizzlepattern swz_encode, int threadcount)
{
	int xsize = input_image->xsize;
	int ysize = input_image->ysize;
	int zsize = input_image->zsize;

	int xblocks = (xsize + xdim - 1) / xdim;
	int yblocks = (ysize + ydim - 1) / ydim;
	int zblocks = (zsize + zdim - 1) / zdim;

	uint8_t *buffer = (uint8_t *) malloc(xblocks * yblocks * zblocks * 16);
	if (!buffer)
	{
		printf("Ran out of memory\n");
		exit(1);
	}

	if (!suppress_progress_counter)
		printf("%d blocks to process ..\n", xblocks * yblocks * zblocks);

	encode_astc_image(input_image, NULL, xdim, ydim, zdim, ewp, decode_mode, swz_encode, swz_encode, buffer, 0, threadcount);

	end_coding_time = get_time();

	astc_header hdr;
	hdr.magic[0] = MAGIC_FILE_CONSTANT & 0xFF;
	hdr.magic[1] = (MAGIC_FILE_CONSTANT >> 8) & 0xFF;
	hdr.magic[2] = (MAGIC_FILE_CONSTANT >> 16) & 0xFF;
	hdr.magic[3] = (MAGIC_FILE_CONSTANT >> 24) & 0xFF;
	hdr.blockdim_x = xdim;
	hdr.blockdim_y = ydim;
	hdr.blockdim_z = zdim;
	hdr.xsize[0] = xsize & 0xFF;
	hdr.xsize[1] = (xsize >> 8) & 0xFF;
	hdr.xsize[2] = (xsize >> 16) & 0xFF;
	hdr.ysize[0] = ysize & 0xFF;
	hdr.ysize[1] = (ysize >> 8) & 0xFF;
	hdr.ysize[2] = (ysize >> 16) & 0xFF;
	hdr.zsize[0] = zsize & 0xFF;
	hdr.zsize[1] = (zsize >> 8) & 0xFF;
	hdr.zsize[2] = (zsize >> 16) & 0xFF;

	FILE *wf = fopen(filename, "wb");
	fwrite(&hdr, 1, sizeof(astc_header), wf);
	fwrite(buffer, 1, xblocks * yblocks * zblocks * 16, wf);
	fclose(wf);
	free(buffer);
}



astc_codec_image *pack_and_unpack_astc_image(const astc_codec_image * input_image,
											 int xdim,
											 int ydim,
											 int zdim,
											 const error_weighting_params * ewp, astc_decode_mode decode_mode, swizzlepattern swz_encode, swizzlepattern swz_decode, int bitness, int threadcount)
{
	int xsize = input_image->xsize;
	int ysize = input_image->ysize;
	int zsize = input_image->zsize;

	astc_codec_image *img = allocate_image(bitness, xsize, ysize, zsize, 0);

	/*
	   allocate_output_image_space( bitness, xsize, ysize, zsize ); */

	int xblocks = (xsize + xdim - 1) / xdim;
	int yblocks = (ysize + ydim - 1) / ydim;
	int zblocks = (zsize + zdim - 1) / zdim;

	if (!suppress_progress_counter)
		printf("%d blocks to process...\n", xblocks * yblocks * zblocks);

	encode_astc_image(input_image, img, xdim, ydim, zdim, ewp, decode_mode, swz_encode, swz_decode, NULL, 1, threadcount);

	if (!suppress_progress_counter)
		printf("\n");

	return img;
}


void find_closest_blockdim_2d(float target_bitrate, int *x, int *y, int consider_illegal)
{
	int blockdims[6] = { 4, 5, 6, 8, 10, 12 };

	float best_error = 1000;
	float aspect_of_best = 1;
	int i, j;

	// Y dimension
	for (i = 0; i < 6; i++)
	{
		// X dimension
		for (j = i; j < 6; j++)
		{
			//              NxN       MxN         8x5               10x5              10x6
			int is_legal = (j==i) || (j==i+1) || (j==3 && i==1) || (j==4 && i==1) || (j==4 && i==2);

			if(consider_illegal || is_legal)
			{
				float bitrate = 128.0f / (blockdims[i] * blockdims[j]);
				float bitrate_error = fabs(bitrate - target_bitrate);
				float aspect = (float)blockdims[j] / blockdims[i];
				if (bitrate_error < best_error || (bitrate_error == best_error && aspect < aspect_of_best))
				{
					*x = blockdims[j];
					*y = blockdims[i];
					best_error = bitrate_error;
					aspect_of_best = aspect;
				}
			}
		}
	}
}



void find_closest_blockdim_3d(float target_bitrate, int *x, int *y, int *z, int consider_illegal)
{
	int blockdims[4] = { 3, 4, 5, 6 };

	float best_error = 1000;
	float aspect_of_best = 1;
	int i, j, k;

	for (i = 0; i < 4; i++)	// Z
		for (j = i; j < 4; j++) // Y
			for (k = j; k < 4; k++) // X
			{
				//              NxNxN              MxNxN                  MxMxN
				int is_legal = ((k==j)&&(j==i)) || ((k==j+1)&&(j==i)) || ((k==j)&&(j==i+1));

				if(consider_illegal || is_legal)
				{
					float bitrate = 128.0f / (blockdims[i] * blockdims[j] * blockdims[k]);
					float bitrate_error = fabs(bitrate - target_bitrate);
					float aspect = (float)blockdims[k] / blockdims[j] + (float)blockdims[j] / blockdims[i] + (float)blockdims[k] / blockdims[i];

					if (bitrate_error < best_error || (bitrate_error == best_error && aspect < aspect_of_best))
					{
						*x = blockdims[k];
						*y = blockdims[j];
						*z = blockdims[i];
						best_error = bitrate_error;
						aspect_of_best = aspect;
					}
				}
			}
}


void compare_two_files(const char *filename1, const char *filename2, int low_fstop, int high_fstop, int psnrmode)
{
	int load_result1;
	int load_result2;
	astc_codec_image *img1 = astc_codec_load_image(filename1, 0, &load_result1);
	if (load_result1 < 0)
	{
		printf("Failed to load file %s.\n", filename1);
		exit(1);
	}
	astc_codec_image *img2 = astc_codec_load_image(filename2, 0, &load_result2);
	if (load_result2 < 0)
	{
		printf("Failed to load file %s.\n", filename2);
		exit(1);
	}

	int file1_components = load_result1 & 0x7;
	int file2_components = load_result2 & 0x7;
	int comparison_components = MAX(file1_components, file2_components);

	int compare_hdr = 0;
	if (load_result1 & 0x80)
		compare_hdr = 1;
	if (load_result2 & 0x80)
		compare_hdr = 1;

	compute_error_metrics(compare_hdr, comparison_components, img1, img2, low_fstop, high_fstop, psnrmode);
}


union if32
{
	float f;
	int32_t s;
	uint32_t u;
};


// The ASTC codec is written with the assumption that a float threaded through
// the "if32" union will in fact be stored and reloaded as a 32-bit IEEE-754 single-precision
// float, stored with round-to-nearest rounding. This is always the case in an
// IEEE-754 compliant system, however not every system is actually IEEE-754 compliant
// in the first place. As such, we run a quick test to check that this is actually the case
// (e.g. gcc on 32-bit x86 will typically fail unless -msse2 -mfpmath=sse2 is specified).

volatile float xprec_testval = 2.51f;
void test_inappropriate_extended_precision(void)
{
	if32 p;
	p.f = xprec_testval + 12582912.0f;
	float q = p.f - 12582912.0f;
	if (q != 3.0f)
	{
		printf("Single-precision test failed; please recompile with proper IEEE-754 support.\n");
		exit(1);
	}
}

// Debug routine to dump the entire image if requested.
void dump_image(astc_codec_image * img)
{
	int x, y, z, xdim, ydim, zdim;

	printf("\n\nDumping image ( %d x %d x %d + %d)...\n\n", img->xsize, img->ysize, img->zsize, img->padding);

	if (img->zsize != 1)
		zdim = img->zsize + 2 * img->padding;
	else
		zdim = img->zsize;

	ydim = img->ysize + 2 * img->padding;
	xdim = img->xsize + 2 * img->padding;

	for (z = 0; z < zdim; z++)
	{
		if (z != 0)
			printf("\n\n");
		for (y = 0; y < ydim; y++)
		{
			if (y != 0)
				printf("\n");
			for (x = 0; x < xdim; x++)
			{
				printf("  0x%08X", *(int unsigned *)&img->imagedata8[z][y][x]);
			}
		}
	}
	printf("\n\n");
}

extern "C" DllExport int ConvertImage(cli* ctx);

DllExport int ConvertImage(cli* ctx) {
	int i;

	test_inappropriate_extended_precision();

	// initialization routines
	prepare_angular_tables();
	build_quantization_mode_table();

	start_time = get_time();

	astc_decode_mode decode_mode = ctx->decodeMode;

	const char* input_filename = ctx->infile;
	const char* output_filename = ctx->outfile;

	int array_size = 1;

	int silentmode = 0;
	int timemode = 0;
	int psnrmode = 0;

	error_weighting_params ewp;

	ewp.rgb_power = 1.0f;
	ewp.alpha_power = 1.0f;
	ewp.rgb_base_weight = 1.0f;
	ewp.alpha_base_weight = 1.0f;
	ewp.rgb_mean_weight = 0.0f;
	ewp.rgb_stdev_weight = 0.0f;
	ewp.alpha_mean_weight = 0.0f;
	ewp.alpha_stdev_weight = 0.0f;

	ewp.rgb_mean_and_stdev_mixing = 0.0f;
	ewp.mean_stdev_radius = 0;
	ewp.enable_rgb_scale_with_alpha = 0;
	ewp.alpha_radius = 0;

	ewp.block_artifact_suppression = 0.0f;
	ewp.rgba_weights[0] = 1.0f;
	ewp.rgba_weights[1] = 1.0f;
	ewp.rgba_weights[2] = 1.0f;
	ewp.rgba_weights[3] = 1.0f;
	ewp.ra_normal_angular_scale = 0;

	swizzlepattern swz_encode = { 0, 1, 2, 3 };
	swizzlepattern swz_decode = { 0, 1, 2, 3 };


	int thread_count = 0;		// default value
	int thread_count_autodetected = 0;

	int preset_has_been_set = 0;

	int plimit_autoset = -1;
	int plimit_user_specified = -1;
	int plimit_set_by_user = 0;

	float dblimit_autoset_2d = 0.0;
	float dblimit_autoset_3d = 0.0;
	float dblimit_user_specified = 0.0;
	int dblimit_set_by_user = 0;

	float oplimit_autoset = 0.0;
	float oplimit_user_specified = 0.0;
	int oplimit_set_by_user = 0;

	float mincorrel_autoset = 0.0;
	float mincorrel_user_specified = 0.0;
	int mincorrel_set_by_user = 0;

	float bmc_user_specified = 0.0;
	float bmc_autoset = 0.0;
	int bmc_set_by_user = 0;

	int maxiters_user_specified = 0;
	int maxiters_autoset = 0;
	int maxiters_set_by_user = 0;

	int pcdiv = 1;

	int xdim = -1;
	int ydim = -1;
	int zdim = -1;

	int target_bitrate_set = 0;
	float target_bitrate = 0;

	int print_block_mode_histogram = 0;

	float log10_texels_2d = 0.0f;
	float log10_texels_3d = 0.0f;

	int low_fstop = -10;
	int high_fstop = 10;

	switch (ctx->blockSize)
	{
	case ASTC4x4: xdim = 4; ydim = 4; zdim = 1; break;
	case ASTC5x4: xdim = 5; ydim = 4; zdim = 1; break;
	case ASTC5x5: xdim = 5; ydim = 5; zdim = 1; break;
	case ASTC6x5: xdim = 6; ydim = 5; zdim = 1; break;
	case ASTC6x6: xdim = 6; ydim = 6; zdim = 1; break;
	case ASTC8x5: xdim = 8; ydim = 5; zdim = 1; break;
	case ASTC8x6: xdim = 8; ydim = 6; zdim = 1; break;
	case ASTC8x8: xdim = 8; ydim = 8; zdim = 1; break;
	case ASTC10x5: xdim = 10; ydim = 5; zdim = 1; break;
	case ASTC10x6: xdim = 10; ydim = 6; zdim = 1; break;
	case ASTC10x8: xdim = 10; ydim = 8; zdim = 1; break;
	case ASTC10x10: xdim = 10; ydim = 10; zdim = 1; break;
	case ASTC12x10: xdim = 12; ydim = 10; zdim = 1; break;
	case ASTC12x12: xdim = 12; ydim = 12; zdim = 1; break;

	case ASTC3x3x3:xdim = 3; ydim = 3; zdim = 3; break;
	case ASTC4x3x3:xdim = 4; ydim = 3; zdim = 3; break;
	case ASTC4x4x3:xdim = 4; ydim = 4; zdim = 3; break;
	case ASTC4x4x4:xdim = 4; ydim = 4; zdim = 4; break;
	case ASTC5x4x4:xdim = 5; ydim = 4; zdim = 4; break;
	case ASTC5x5x4:xdim = 5; ydim = 5; zdim = 4; break;
	case ASTC5x5x5:xdim = 5; ydim = 5; zdim = 5; break;
	case ASTC6x5x5:xdim = 6; ydim = 5; zdim = 5; break;
	case ASTC6x6x5:xdim = 6; ydim = 6; zdim = 5; break;
	case ASTC6x6x6:xdim = 6; ydim = 6; zdim = 6; break;
	default:
		break;
	}

	switch (ctx->speedMode) {
	case VeryFast:
		plimit_autoset = 2;
		oplimit_autoset = 1.0;
		dblimit_autoset_2d = MAX(70 - 35 * log10_texels_2d, 53 - 19 * log10_texels_2d);
		dblimit_autoset_3d = MAX(70 - 35 * log10_texels_3d, 53 - 19 * log10_texels_3d);
		bmc_autoset = 25;
		mincorrel_autoset = 0.5;
		maxiters_autoset = 1;

		switch (ydim)
		{
		case 4:
			pcdiv = 240;
			break;
		case 5:
			pcdiv = 56;
			break;
		case 6:
			pcdiv = 64;
			break;
		case 8:
			pcdiv = 47;
			break;
		case 10:
			pcdiv = 36;
			break;
		case 12:
			pcdiv = 30;
			break;
		default:
			pcdiv = 30;
			break;
		}
		break;
	case Fast:
		plimit_autoset = 4;
		oplimit_autoset = 1.0;
		mincorrel_autoset = 0.5;
		dblimit_autoset_2d = MAX(85 - 35 * log10_texels_2d, 63 - 19 * log10_texels_2d);
		dblimit_autoset_3d = MAX(85 - 35 * log10_texels_3d, 63 - 19 * log10_texels_3d);
		bmc_autoset = 50;
		maxiters_autoset = 1;


		switch (ydim)
		{
		case 4:
			pcdiv = 60;
			break;
		case 5:
			pcdiv = 27;
			break;
		case 6:
			pcdiv = 30;
			break;
		case 8:
			pcdiv = 24;
			break;
		case 10:
			pcdiv = 16;
			break;
		case 12:
			pcdiv = 20;
			break;
		default:
			pcdiv = 20;
			break;
		};
		break;
	case Medium:
		plimit_autoset = 25;
		oplimit_autoset = 1.2f;
		mincorrel_autoset = 0.75f;
		dblimit_autoset_2d = MAX(95 - 35 * log10_texels_2d, 70 - 19 * log10_texels_2d);
		dblimit_autoset_3d = MAX(95 - 35 * log10_texels_3d, 70 - 19 * log10_texels_3d);
		bmc_autoset = 75;
		maxiters_autoset = 2;

		switch (ydim)
		{
		case 4:
			pcdiv = 25;
			break;
		case 5:
			pcdiv = 15;
			break;
		case 6:
			pcdiv = 15;
			break;
		case 8:
			pcdiv = 10;
			break;
		case 10:
			pcdiv = 8;
			break;
		case 12:
			pcdiv = 6;
			break;
		default:
			pcdiv = 6;
			break;
		};
		break;
	case Thorough:
		plimit_autoset = 100;
		oplimit_autoset = 2.5f;
		mincorrel_autoset = 0.95f;
		dblimit_autoset_2d = MAX(105 - 35 * log10_texels_2d, 77 - 19 * log10_texels_2d);
		dblimit_autoset_3d = MAX(105 - 35 * log10_texels_3d, 77 - 19 * log10_texels_3d);
		bmc_autoset = 95;
		maxiters_autoset = 4;

		switch (ydim)
		{
		case 4:
			pcdiv = 12;
			break;
		case 5:
			pcdiv = 7;
			break;
		case 6:
			pcdiv = 7;
			break;
		case 8:
			pcdiv = 5;
			break;
		case 10:
			pcdiv = 4;
			break;
		case 12:
			pcdiv = 3;
			break;
		default:
			pcdiv = 3;
			break;
		};
		break;
	case Exhaustive:
		plimit_autoset = PARTITION_COUNT;
		oplimit_autoset = 1000.0f;
		mincorrel_autoset = 0.99f;
		dblimit_autoset_2d = 999.0f;
		dblimit_autoset_3d = 999.0f;
		bmc_autoset = 100;
		maxiters_autoset = 4;

		switch (ydim)
		{
		case 4:
			pcdiv = 3;
			break;
		case 5:
			pcdiv = 1;
			break;
		case 6:
			pcdiv = 1;
			break;
		case 8:
			pcdiv = 1;
			break;
		case 10:
			pcdiv = 1;
			break;
		case 12:
			pcdiv = 1;
			break;
		default:
			pcdiv = 1;
			break;
		}
		break;
	default:
		break;
	}

	if (ctx->method == Compare) {
		compare_two_files(input_filename, output_filename, low_fstop, high_fstop, psnrmode);
		return 0;
	}

	float texel_avg_error_limit_2d = 0.0f;
	float texel_avg_error_limit_3d = 0.0f;

	//Setup encoding env
	if (ctx->method == Compression || ctx->method == DoBoth) {
		progress_counter_divider = pcdiv;

		int partitions_to_test = plimit_set_by_user ? plimit_user_specified : plimit_autoset;
		float dblimit_2d = dblimit_set_by_user ? dblimit_user_specified : dblimit_autoset_2d;
		float dblimit_3d = dblimit_set_by_user ? dblimit_user_specified : dblimit_autoset_3d;
		float oplimit = oplimit_set_by_user ? oplimit_user_specified : oplimit_autoset;
		float mincorrel = mincorrel_set_by_user ? mincorrel_user_specified : mincorrel_autoset;

		int maxiters = maxiters_set_by_user ? maxiters_user_specified : maxiters_autoset;
		ewp.max_refinement_iters = maxiters;

		ewp.block_mode_cutoff = (bmc_set_by_user ? bmc_user_specified : bmc_autoset) / 100.0f;

		if (rgb_force_use_of_hdr == 0)
		{
			texel_avg_error_limit_2d = pow(0.1f, dblimit_2d * 0.1f) * 65535.0f * 65535.0f;
			texel_avg_error_limit_3d = pow(0.1f, dblimit_3d * 0.1f) * 65535.0f * 65535.0f;
		}
		else
		{
			texel_avg_error_limit_2d = 0.0f;
			texel_avg_error_limit_3d = 0.0f;
		}
		ewp.partition_1_to_2_limit = oplimit;
		ewp.lowest_correlation_cutoff = mincorrel;

		if (partitions_to_test < 1)
			partitions_to_test = 1;
		else if (partitions_to_test > PARTITION_COUNT)
			partitions_to_test = PARTITION_COUNT;
		ewp.partition_search_limit = partitions_to_test;

		// if diagnostics are run, force the thread count to 1.
		if (
#ifdef DEBUG_PRINT_DIAGNOSTICS
			diagnostics_tile >= 0 ||
#endif
			print_tile_errors > 0 || print_statistics > 0)
		{
			thread_count = 1;
			thread_count_autodetected = 0;
		}

		if (thread_count < 1)
		{
			thread_count = get_number_of_cpus();
			thread_count_autodetected = 1;
		}


		// Specifying the error weight of a color component as 0 is not allowed.
		// If weights are 0, then they are instead set to a small positive value.

		float max_color_component_weight = MAX(MAX(ewp.rgba_weights[0], ewp.rgba_weights[1]),
			MAX(ewp.rgba_weights[2], ewp.rgba_weights[3]));
		ewp.rgba_weights[0] = MAX(ewp.rgba_weights[0], max_color_component_weight / 1000.0f);
		ewp.rgba_weights[1] = MAX(ewp.rgba_weights[1], max_color_component_weight / 1000.0f);
		ewp.rgba_weights[2] = MAX(ewp.rgba_weights[2], max_color_component_weight / 1000.0f);
		ewp.rgba_weights[3] = MAX(ewp.rgba_weights[3], max_color_component_weight / 1000.0f);
	}

	int padding = MAX(ewp.mean_stdev_radius, ewp.alpha_radius);

	// determine encoding bitness as follows:
	// if enforced by the output format, follow the output format's result
	// else use decode_mode to pick bitness.
	int bitness = get_output_filename_enforced_bitness(output_filename);
	if (bitness == -1)
	{
		bitness = (decode_mode == DECODE_HDR) ? 16 : 8;
	}

	// Temporary image array (for merging multiple 2D images into one 3D image).
	int* load_results = NULL;
	astc_codec_image** input_images = NULL;

	int load_result = 0;
	astc_codec_image* input_image = NULL;
	astc_codec_image* output_image = NULL;
	int input_components = 0;

	int input_image_is_hdr = 0;

	//Encoding
	if (ctx->method == Compression || ctx->method == DoBoth) {
		// Allocate arrays for image data and load results.
		load_results = new int[array_size];
		input_images = new astc_codec_image * [array_size];

		// Iterate over all input images.
		for (int image_index = 0; image_index < array_size; image_index++)
		{
			// 2D input data.
			if (array_size == 1)
			{
				input_images[image_index] = astc_codec_load_image(input_filename, padding, &load_results[image_index]);
			}

			// 3D input data - multiple 2D images.
			else
			{
				char new_input_filename[256];

				// Check for extension: <name>.<extension>
				if (NULL == strrchr(input_filename, '.'))
				{
					printf("Unable to determine file type from extension: %s\n", input_filename);
					exit(1);
				}

				// Construct new file name and load: <name>_N.<extension>
				strcpy(new_input_filename, input_filename);
				sprintf(strrchr(new_input_filename, '.'), "_%d%s", image_index, strrchr(input_filename, '.'));
				input_images[image_index] = astc_codec_load_image(new_input_filename, padding, &load_results[image_index]);

				// Check image is not 3D.
				if (input_images[image_index]->zsize != 1)
				{
					printf("3D source images not supported with -array option: %s\n", new_input_filename);
					exit(1);
				}

				// BCJ(DEBUG)
				// printf("\n\n Image %d \n", image_index);
				// dump_image( input_images[image_index] );
				// printf("\n\n");
			}

			// Check load result.
			if (load_results[image_index] < 0)
			{
				printf("Failed to load image %s\n", input_filename);
				exit(1);
			}

			// Check format matches other slices.
			if (load_results[image_index] != load_results[0])
			{
				printf("Mismatching image format - image 0 and %d are a different format\n", image_index);
				exit(1);
			}
		}

		load_result = load_results[0];

		// Assign input image.
		if (array_size == 1)
		{
			input_image = input_images[0];
		}

		// Merge input image data.
		else
		{
			int i, z, xsize, ysize, zsize, bitness, slice_size;

			xsize = input_images[0]->xsize;
			ysize = input_images[0]->ysize;
			zsize = array_size;
			bitness = (load_result & 0x80) ? 16 : 8;
			slice_size = (xsize + (2 * padding)) * (ysize + (2 * padding));

			// Allocate image memory.
			input_image = allocate_image(bitness, xsize, ysize, zsize, padding);

			// Combine 2D source images into one 3D image (skip padding slices as these don't exist in 2D textures).
			for (z = padding; z < zsize + padding; z++)
			{
				if (bitness == 8)
				{
					memcpy(*input_image->imagedata8[z], *input_images[z - padding]->imagedata8[0], slice_size * 4 * sizeof(uint8_t));
				}
				else
				{
					memcpy(*input_image->imagedata16[z], *input_images[z - padding]->imagedata16[0], slice_size * 4 * sizeof(uint16_t));
				}
			}

			// Clean up temporary images.
			for (i = 0; i < array_size; i++)
			{
				destroy_image(input_images[i]);
			}
			input_images = NULL;

			// Clamp texels outside the actual image area.
			fill_image_padding_area(input_image);

			// BCJ(DEBUG)
			// dump_image( input_image );
		}

		input_components = load_result & 7;
		input_image_is_hdr = (load_result & 0x80) ? 1 : 0;

		if (input_image->zsize > 1)
		{
			ewp.texel_avg_error_limit = texel_avg_error_limit_3d;
		}
		else
		{
			ewp.texel_avg_error_limit = texel_avg_error_limit_2d;
		}
		expand_block_artifact_suppression(xdim, ydim, zdim, &ewp);

		if (padding > 0 || ewp.rgb_mean_weight != 0.0f || ewp.rgb_stdev_weight != 0.0f || ewp.alpha_mean_weight != 0.0f || ewp.alpha_stdev_weight != 0.0f)
		{
			if (!silentmode)
			{
				printf("Computing texel-neighborhood means and variances ... ");
				fflush(stdout);
			}
			compute_averages_and_variances(input_image, ewp.rgb_power, ewp.alpha_power, ewp.mean_stdev_radius, ewp.alpha_radius, swz_encode);
			if (!silentmode)
			{
				printf("done\n");
				fflush(stdout);
			}
		}
	}

	//Decoding
	if (ctx->method == Decompression)
		output_image = load_astc_file(input_filename, bitness, decode_mode, swz_decode);
	else if (ctx->method == DoBoth)
		output_image = pack_and_unpack_astc_image(input_image, xdim, ydim, zdim, &ewp, decode_mode, swz_encode, swz_decode, bitness, thread_count);

	// store image
	if (ctx->method == Decompression || ctx->method == DoBoth)
	{
		int store_result = -1;
		const char* format_string = "";

		store_result = astc_codec_store_image(output_image, output_filename, bitness, &format_string);

		if (store_result < 0)
		{
			printf("Failed to store image %s\n", output_filename);
			exit(1);
		}
		else
		{
			if (!silentmode)
			{
				printf("Stored %s image %s with %d color channels\n", format_string, output_filename, store_result);
			}
		}
	}
	if (ctx->method == Compression)
	{
		store_astc_file(input_image, output_filename, xdim, ydim, zdim, &ewp, decode_mode, swz_encode, thread_count);
	}

	return 0;
};