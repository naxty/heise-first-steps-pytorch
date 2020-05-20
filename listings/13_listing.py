example_input = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(fruit_model, example_input)
traced_script_module.save("fruit_model.pt")