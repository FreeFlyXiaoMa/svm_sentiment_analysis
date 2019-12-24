import tensorflow as tf

# tf.app.flags.DEFINE_string("model_type", "charcnn", "默认为cnn")
# tf.app.flags.DEFINE_string("sentence", "微信可以登录吗", "默认为cnn")
# FLAGS = tf.app.flags.FLAGS
model_type = 'textcnn'
sentences = '微信可以登录吗'

infer = None
if model_type == 'textcnn':
    import model.textcnn.Infer as textcnn_infer
    infer = textcnn_infer.Infer()
elif model_type == 'charcnn':
    import model.char_cnn.Infer as char_cnn_infer
    infer = char_cnn_infer.Infer()
elif model_type == 'fasttext':
    import model.fast_text.Infer as fasttext_infer
    infer = fasttext_infer.Infer()
elif model_type == 'textrnn':
    import model.textrnn.Infer as textrnn_infer
    infer = textrnn_infer.Infer()
elif model_type == 'birnn_attention':
    import model.birnn_attention.Infer as birnn_attention_infer
    infer = birnn_attention_infer.Infer()
elif model_type == 'leam':
    import model.leam.Infer as leam_infer
    infer = leam_infer.Infer()
elif model_type == 'transformer':
    import model.transformer.Infer as transformer_infer
    infer = transformer_infer.Infer()
else:
    print("do not exist this model")
print(infer.infer([sentences]))