def input_fn(fns = ['../data/tfrecords/train_1.tfrecords'],
             sz=128, nt=10, aug_tfms=aug_tfms,
             stats_fn='stat.csv', stats_sep=','):
    """
    Create tf.data.Iterator from tfrecord file.
    
    Receives: 
        fns = a list of tfrecords files
    
    """
    dataset = tf.data.TFRecordDataset(fns)
    
    y = tf.zeros([bs, 1])
    def parser_train(serialized_example):
        # experimental. TODO: read only needed samples
        shape = (61, 501, 501, 3)
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
        return x, y
    
    stats = np.fromfile(stats_fn, sep=stats_sep)
    tfms, _ = tfms_from_stats(stats, sz, aug_tfms=aug_tfms, crop_type=CropType.NO)
    
    dataset = dataset.map(parser_train)
    dataset = dataset.map(tfms)
    dataset = dataset.batch(bs)
    dataset = dataset.repeat()
    # dataset = dataset.prefetch()
    
#     return dataset
    
    y = tf.zeros([bs, 1])
    iterator = dataset.make_one_shot_iterator()
    x, _ = iterator.get_next()
    return x, y