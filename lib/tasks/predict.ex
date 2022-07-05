defmodule Mix.Tasks.Predict do
  use Mix.Task
  require Axon

  @requirements ["app.start"]

  alias MNIST

  @doc """
  Example:
  mix predict --image "/home/path/dev/cool/priv/images/0.png"
  """
  def run(args) do
    {[image: image_path], _} = OptionParser.parse!(args, strict: [image: :string])

    {:ok, image} =
      Image.open!(image_path)
      |> Image.resize!(28)
      |> Image.to_colorspace!(:grey16)
      |> Image.to_nx()

    image =
      image
      |> Nx.reduce_min(axes: [2])
      |> Nx.reshape({1, 28, 28})
      |> List.wrap()
      |> Nx.stack()

    model = MNIST.Model.new({1, 28, 28})
    params = MNIST.Model.load!()

    prediction = MNIST.Model.predict(model, params, image)

    IO.ANSI.blink_slow()

    Mix.Shell.IO.info(
      IO.ANSI.blue_background() <>
        "Prediction: #{prediction}." <>
        IO.ANSI.reset()
    )
  end
end
