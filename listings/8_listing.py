def test(model, test_loader, loss_function, device="cpu"):
    model.eval()
    batch_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            predictions = model(data)
            batch_loss += loss_function(predictions, target).item()
            predicted_labels = predictions.argmax(dim=1, keepdim=True)
            correct += predicted_labels.eq(target.view_as(predicted_labels)).sum().item()
    average_loss = batch_loss / len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Average loss: {average_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.3f}%)\n')
test(model, test_loader, loss_function)