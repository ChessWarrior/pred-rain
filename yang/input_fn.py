from utils.imports import *
from utils.transforms import *
from cvt2tfrecord import fn_record_to_count

def input_fn(bs, sz, nt, aug_tfms, fns,
             stats_fn='stat.csv', stats_sep=',', num_parallel_calls=8):
    """
    TODO: data interleave
    Create tf.data.Iterator from tfrecord file.
    
    Receives: 
        fns = a list of tfrecords files **without suffix**
        sz  = the dimension the input images are resized to (int)
        aug_tfms = a list of transforms to apply to input images
    """
    fn_records = [o + '.tfrecords' for o in fns]
    dataset = tf.data.TFRecordDataset(fn_records)
    
    def parser_train(serialized_example):
        # experimental. TODO: read only needed samples
        shape = (nt, 501, 501, 3)
        context_features = {
                'time_stamp': tf.FixedLenFeature([], tf.string),
            }
        sequence_features = {
                "raw_png": tf.FixedLenSequenceFeature([], dtype=tf.string)
            }
        
        features, sequence_features = tf.parse_single_sequence_example(
            serialized_example, 
            context_features=context_features, 
            sequence_features=sequence_features)

        x = tf.map_fn(tf.image.decode_png, sequence_features['raw_png'], dtype=tf.uint8,
                    back_prop=False, swap_memory=False, infer_shape=False)
        x = tf.cast(x, tf.float32)
        x /= 255
        x.set_shape(shape)
        x = tf.expand_dims(x[:,:,:,0], axis=3)
        return x 
    
    stats = np.fromfile(stats_fn, sep=stats_sep) # normalization stat
    tfms, _ = tfms_from_stats(stats, sz, aug_tfms=aug_tfms, crop_type=CropType.NO)
    
    dataset = dataset.map(parser_train, num_parallel_calls=num_parallel_calls)
    dataset = dataset.map(tfms, num_parallel_calls=num_parallel_calls)
    dataset = dataset.batch(bs)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(2)
    
    y = tf.zeros([bs, 1])
    iterator = dataset.make_one_shot_iterator()
    x = iterator.get_next()
    
    fn_counts = [fn_record_to_count(o) for o in fns]
    num_iterations = 0
    for fn_count in fn_counts:
        with open(fn_count, 'r') as f:
            num_iterations+= int(f.read())
            
    return x, y, num_iterations