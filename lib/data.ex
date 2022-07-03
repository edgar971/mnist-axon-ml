defmodule MNIST.Data do
  @moduledoc """
  Documentation for `MNIST.Data`.
  """

  def download_mnist() do
    Scidata.MNIST.download(base_dir: Path.join(Application.app_dir(:mnist_axon, "priv"), "data"))
  end

  def transform_images({binary, type, shape}) do
    binary
    |> Nx.from_binary(type)
    |> Nx.reshape(shape)
    |> Nx.divide(255)
  end

  def transform_labels({binary, type, _}) do
    binary
    |> Nx.from_binary(type)
    |> Nx.new_axis(-1)
    |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
  end

  def get_train_test_val_dataset(train_split, val_split) do
    {images, labels} = download_mnist()

    images =
      images
      |> transform_images()
      |> Nx.to_batched_list(64)

    labels =
      labels
      |> transform_labels()
      |> Nx.to_batched_list(64)

    data = Enum.zip(images, labels)

    training_count = floor(train_split * Enum.count(data))
    validation_count = floor(val_split * training_count)

    {training_data, test_data} = Enum.split(data, training_count)
    {validation_data, training_data} = Enum.split(training_data, validation_count)

    {training_data, test_data, validation_data}
  end
end
