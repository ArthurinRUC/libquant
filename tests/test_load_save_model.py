from libquant import load_quant_model, quant, save_quant_model


def test_load_save_model(model_path, save_path):
    model = load_quant_model(model_path)
    quant(model, nbits=4)
    save_quant_model(model, save_path)
