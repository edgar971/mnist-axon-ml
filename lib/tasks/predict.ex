defmodule Mix.Tasks.Predict do
  use Mix.Task
  require Axon

  @requirements ["app.start"]

  alias MNIST

  def run(_) do
    EXLA.set_as_nx_default([:tpu, :cuda, :rocm, :host])
    [image, label] = MNIST.Data.get_random_image()
    model = MNIST.Model.new({1, 28, 28})

    Mix.Shell.IO.info("loading saved model...")
    params = MNIST.Model.load!()

    prediction = MNIST.Model.predict(model, params, image)

    Mix.Shell.IO.info("Pred: #{prediction}. Label: #{label} \n")
  end
end
