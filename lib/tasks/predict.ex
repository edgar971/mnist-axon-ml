defmodule Mix.Tasks.Predict do
  use Mix.Task
  require Axon

  @requirements ["app.start"]

  alias MNIST

  def run(_) do
    EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])
    {_, test, _} = MNIST.Data.get_train_test_val_dataset(0.8, 0.2)

    model = MNIST.Model.new({1, 28, 28})

    Mix.Shell.IO.info("loading saved model...")
    params = MNIST.Model.load!()

    MNIST.Model.test(model, params, test)

    :ok
  end
end
