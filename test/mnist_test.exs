defmodule MNISTTest do
  use ExUnit.Case
  doctest MNIST

  test "greets the world" do
    assert MNIST.hello() == :world
  end
end
