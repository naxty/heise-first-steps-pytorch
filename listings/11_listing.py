train_size = int(0.8 * len(image_dataset))
test_size = len(image_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(image_dataset, [train_size, test_size])

train(fruit_model, train_loader, optimizer, loss_function, device=device)
test(fruit_model, test_loader, loss_function)
