defmodule MNIST.Model do
  @moduledoc """
  Documentation for `MNIST.Model`.
  """

  def new({channels, height, width}) do
    Axon.input({nil, channels, height, width}, "input")
    |> Axon.flatten()
    |> Axon.dense(64, activation: :relu)
    |> Axon.dense(10)
    |> Axon.activation(:softmax)
  end

  def train(model, train_data, validation_data) do
    model
    |> Axon.Loop.trainer(:categorical_cross_entropy, Axon.Optimizers.adamw(0.005))
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.validate(model, validation_data)
    |> Axon.Loop.run(train_data, %{}, compiler: EXLA, epochs: 10)
  end

  def test(model, params, test_data) do
    model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(:accuracy, "Accuracy")
    |> Axon.Loop.run(test_data, params)
  end

  def save!(params) do
    content = Nx.serialize(params)

    File.write!(path(), content)
  end

  def load! do
    path()
    |> File.read!()
    |> Nx.deserialize()
  end

  def path do
    Path.join(Application.app_dir(:mnist_axon, "priv"), "model.axon")
  end
end
