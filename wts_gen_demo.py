
import os
import numpy as np
import pickle
import struct

### conv
# the weights are specified as a contiguous array in GKCRS order, where G is the number of groups, K the number of output feature maps, C the number of input channels, and R and S are the height and width of the filter
## fc
# set the kernel weights. The expected format is an array of KC values, where K is the number of outputs and C is the number of inputs. 
#G = 1


#precision = 'full'
precision = 'half'

assert(precision in ['full', 'half'])
if precision == 'full':
    datatype = np.float32
    datatype_id = 0
elif precision == 'half':
    datatype = np.float16
#    datatype = np.float32
    datatype_id = 1
    
flatten_order = 'C'
transpose_order_conv = (3, 2, 0, 1)
transpose_order_fc = (1, 0)


def make_printable32(in_array):
    
    hex_array = [float_to_hex(x)[2:] for x in in_array]
    
    return ' '.join(hex_array)

def make_printable16(in_array):
    
    hex_array = [float_to_hex(x)[2:6] for x in in_array]
    
    return ' '.join(hex_array)

def make_printable32_plain(in_array):
    
    float_array = [str(x) for x in in_array]
    
    return ' '.join(float_array)

def make_printable16_plain(in_array):
    
    float_array = [str(x) for x in in_array]
    
    return ' '.join(float_array)


make_printable = {'full': make_printable32, 'half': make_printable16}
make_printable_plain = {'full': make_printable32_plain, 'half': make_printable16_plain}


def float_to_hex(f):
    
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])


if __name__ == '__main__':
    
    data_path = './data'
    in_file = os.path.join(data_path, 'weights_demo.p')
    out_file = os.path.join(data_path, 'weights_demo16_v2.wts')
    weights_data = pickle.load(open(in_file, 'rb'))
    
    with open(out_file, 'w') as outfile:
    
        # first line, number of weight tensors to store
        outfile.write('%s\n' % len(weights_data.keys()) )
    
        # iterate across weight tensors
        for ll in range(1,2):
            
            # get weight name
            w_id = 'conv%d_w' % ll
            b_id = 'conv%d_b' % ll
            
            # retrieve values
            w = weights_data[w_id]
            b = weights_data[b_id]
            
            # adjust to RT format and convert
            w = np.transpose(w, transpose_order_conv)
            w = w.flatten(flatten_order)
            w = w.astype(datatype)
            b = b.astype(datatype)
            
            # line format: <name> <datatype> <size> <hex values> 
            w_line = w_id + (' %d %d ' % (datatype_id, w.size)) + make_printable[precision](w) + '\n'
            b_line = b_id + (' %d %d ' % (datatype_id, b.size)) + make_printable[precision](b) + '\n'
            
            # append to file
            outfile.write(b_line)
            outfile.write(w_line)

