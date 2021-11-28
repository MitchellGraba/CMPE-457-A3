# Image compression
#
# You'll need Python 3 and the netpbm library, but netpbm is provided
# with the assignment code.    You can run this *only* on PNM images,
# which the netpbm library is used for.    You can also display a PNM
# image using the netpbm library as, for example:
#
#     python netpbm.py images/cortex.pnm
#
# The NumPy library should be installed and used for its faster array
# manipulation.  DO NOT USE NUMPY OTHER THAN TO CREATE AND ACCESS
# ARRAYS.  DOING SO WILL LOSE MARKS.


import sys, os, math, time, struct, netpbm
import numpy as np

# Text at the beginning of the compressed file, to identify the codec
# and codec version.

headerText = 'my compressed image - v1.0'


# Compress an image


def compress(inputFile, outputFile):
    # Read the input file into a numpy array of 8-bit values
    #
    # The img.shape is a 3-type with rows,columns,channels, where
    # channels is the number of component in each pixel.  The
    # img.dtype is 'uint8', meaning that each component is an 8-bit
    # unsigned integer.

    img = netpbm.imread(inputFile).astype('uint8')

    # Compress the image
    #
    # REPLACE THIS WITH YOUR OWN CODE TO FILL THE 'outputBytes' ARRAY.
    #
    # Note that single-channel images will have a 'shape' with only two
    # components: the y dimensions and the x dimension.    So you will
    # have to detect this and set the number of channels accordingly.
    # Furthermore, single-channel images must be indexed as img[y,x]
    # instead of img[y,x,1].  You'll need two pieces of similar code:
    # one piece for the single-channel case and one piece for the
    # multi-channel (R,G,B) case.
    #
    # You will build up bytes-strings of 16-bit integers during the
    # encoding.  To convert a 16-bit integer, i, to a byte-string, use
    #
    #   struct.pack('>h', i)
    #
    # where '>' means big-endian and 'h' means 2-byte signed integer.
    # If you know that the integers are unsigned, you should instead
    # use '>H'.
    #
    # Use these byte-strings (and concatenations of these byte-strings
    # when you have multiple integers in sequence) as dictionary keys.
    # DO NOT USE ARRAYS OF INTEGERS AS DICTIONARY KEYS.  DOING SO WILL
    # LOSE MARKS.

    startTime = time.time()

    val_list = map(str, range(-255, 256))
    count = len(range(-255, 256))
    d = dict(zip(val_list, range(512)))

    s = ""

    outputBytes = bytearray()

    diff = []

    # single channel image
    if len(img.shape) == 2:
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                # predictive encoding by taking difference of previous pixels
                if y == 0:
                    val = img[y, x]
                else:
                    val = int(img[y, x]) - int(img[y - 1, x])
                val = str(val)
                diff.append(val)

    # multi-channel image
    else:
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                for c in range(img.shape[2]):
                    # predictive encoding
                    if y == 0:
                        val = img[y, x, c]
                    else:
                        val = int(img[y, x, c]) - int(img[y - 1, x, c])
                    val = str(val)
                    diff.append(val)

    # first pixel exists in dictionary
    s += diff[0]
    for i in range(1, len(diff)):
        # LZW compression
        # LZW_dict keys - subsequence, string
        # LZW_dict values - index, int
        tmp_s = s + "|" + diff[i]
        if tmp_s in d:
            s = tmp_s
        else:
            # convert integer to bytes
            val_b = struct.pack(">H", d[s])
            # append bytes to output array
            outputBytes += val_b
            # append to dictionary if not full
            if count <= 0xFFFE:
                d[tmp_s] = count
                count += 1
            s = diff[i]
    # output last byte
    val_b = struct.pack(">H", d[s])
    # append bytes to output array
    outputBytes += val_b

    endTime = time.time()

    # Output the bytes
    #
    # Include the 'headerText' to identify the type of file.    Include
    # the rows, columns, channels so that the image shape can be
    # reconstructed.

    outputFile.write(('%s\n' % headerText).encode())
    if len(img.shape) == 2:
        outputFile.write(('%d %d\n' % (img.shape[0], img.shape[1])).encode())
    else:
        outputFile.write(('%d %d %d\n' % (img.shape[0], img.shape[1], img.shape[2])).encode())

    outputFile.write(outputBytes)

    # Print information about the compression

    if len(img.shape) == 2:
        inSize = img.shape[0] * img.shape[1]
    else:
        inSize = img.shape[0] * img.shape[1] * img.shape[2]
    outSize = len(outputBytes)

    sys.stderr.write('Input size:         %d bytes\n' % inSize)
    sys.stderr.write('Output size:        %d bytes\n' % outSize)
    sys.stderr.write('Compression factor: %.2f\n' % (inSize / float(outSize)))
    sys.stderr.write('Compression time:   %.2f seconds\n' % (endTime - startTime))


# Uncompress an image

def uncompress(inputFile, outputFile):
    # Check that it's a known file

    if inputFile.readline().decode() != headerText + '\n':
        sys.stderr.write("Input is not in the '%s' format.\n" % headerText)
        sys.exit(1)

    # Read the rows, columns, and channels.
    try:
        RowColChan = [int(x) for x in inputFile.readline().decode().split()]
        rows = RowColChan[0]
        columns = RowColChan[1]
        numChannels = RowColChan[2]  # exception caused here if only a single channel image
    except:
        numChannels = 1  # if it's only a single channel image

    # Read the raw bytes.
    inputBytes = bytearray(inputFile.read())

    # Build the image
    #
    # REPLACE THIS WITH YOUR OWN CODE TO CONVERT THE 'inputBytes'
    # ARRAY INTO AN IMAGE IN 'img'.
    #
    # When unpacking an UNSIGNED 2-byte integer from the inputBytes
    # byte-string, use struct.unpack( '>H', inputBytes[i:i+1] ) for
    # the unsigned integer in indices i and i+1.

    startTime = time.time()
    if numChannels == 1:
        img = np.empty([rows, columns], dtype=np.uint8)
    else:
        img = np.empty([rows, columns, numChannels], dtype=np.uint8)

    # convert byte values back to decimal
    byteIter = []
    for i in range(0, len(inputBytes), 2):
        val = struct.unpack(">H", inputBytes[i:i + 2])[0]
        byteIter.append(val)
    byteIter = iter(byteIter)

    img_data = np.zeros((rows * columns * numChannels))
    # initial setup and dictionary
    val_ls = map(str, range(-255, 256))
    count = len(range(-255, 256))
    LZW_dict = dict(zip(range(512), val_ls))

    s = LZW_dict[next(byteIter)]
    s_int = int(s)
    img_data[0] = s_int
    pos = 1

    while True:
        try:
            # next code
            val = next(byteIter)
            # dictionary look up of next code
            if val in LZW_dict:
                t = LZW_dict[val]
            # next code not found in dictionary
            else:
                # s is a string sequence of numbers
                if '|' in s:
                    t = s + "|" + s.split('|')[0]
                # s is a string of a number
                else:
                    t = s + "|" + s

            # t is a string sequence of numbers
            if '|' in t:
                t_arr = t.split('|')
                for item in t_arr:
                    img_data[pos] = int(item)
                    pos += 1
                LZW_dict[count] = s + "|" + t_arr[0]
            # t is a string of a number
            else:
                t_int = int(t)
                img_data[pos] = t_int
                LZW_dict[count] = s + "|" + t
                pos += 1

            count += 1
            s = t

        # end of python iter is indicated by an exception
        # so we know all codes have been processed when it is thrown
        except:
            break

    img_iter = iter(img_data)

    # convert to img array
    if numChannels == 1:
        for y in range(rows):
            for x in range(columns):
                if y == 0:
                    img[y, x] = next(img_iter)
                else:
                    img[y, x] = next(img_iter) + img[y - 1, x]
    else:
        for y in range(rows):
            for x in range(columns):
                for c in range(numChannels):
                    if y == 0:
                        img[y, x, c] = next(img_iter)
                    else:
                        img[y, x, c] = next(img_iter) + img[y - 1, x, c]

    endTime = time.time()
    sys.stderr.write('Uncompression time %.2f seconds\n' % (endTime - startTime))

    # Output the image

    netpbm.imsave(outputFile, img)


# The command line is
#
#     main.py {flag} {input image filename} {output image filename}
#
# where {flag} is one of 'c' or 'u' for compress or uncompress and
# either filename can be '-' for standard input or standard output.


if len(sys.argv) < 4:
    sys.stderr.write('Usage: main.py c|u {input image filename} {output image filename}\n')
    sys.exit(1)

# Get input file

if sys.argv[2] == '-':
    inputFile = sys.stdin
else:
    try:
        inputFile = open(sys.argv[2], 'rb')
    except:
        sys.stderr.write("Could not open input file '%s'.\n" % sys.argv[2])
        sys.exit(1)

# Get output file

if sys.argv[3] == '-':
    outputFile = sys.stdout
else:
    try:
        outputFile = open(sys.argv[3], 'wb')
    except:
        sys.stderr.write("Could not open output file '%s'.\n" % sys.argv[3])
        sys.exit(1)

# Run the algorithm

if sys.argv[1] == 'c':
    compress(inputFile, outputFile)
elif sys.argv[1] == 'u':
    uncompress(inputFile, outputFile)
else:
    sys.stderr.write('Usage: main.py c|u {input image filename} {output image filename}\n')
    sys.exit(1)
