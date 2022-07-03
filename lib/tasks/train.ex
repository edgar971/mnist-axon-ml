defmodule Mix.Tasks.Train do
  use Mix.Task
  require Axon

  @requirements ["app.start"]

  alias MNIST

  def run(_) do
    EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])

    {train, test, val} = MNIST.Data.get_train_test_val_dataset(0.8, 0.2)

    model = MNIST.Model.new({1, 28, 28})

    Mix.Shell.IO.info("training...")

    state = MNIST.Model.train(model, train, val)

    Mix.Shell.IO.info("testing...")
    MNIST.Model.test(model, state, test)

    Mix.Shell.IO.info("saving...")
    MNIST.Model.save!(state)
    :ok
  end
end
