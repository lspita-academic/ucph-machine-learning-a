// ---------- Page & text setup (article, a4paper, 12pt, a4wide) ----------
#set page(paper: "a4", margin: (x: 2.5cm, y: 2.5cm), numbering: "1")
#set text(size: 12pt, lang: "en")
#set par(justify: true)
#set heading(numbering: "1.1")
#set math.equation(numbering: "(1)")
#set figure(placement: none)

// Colorful links (hyperref)
#show link: set text(fill: blue)
#show ref: set text(fill: blue)

// ---------- Custom commands ----------
// Indicator function: write #ind in math, e.g. $ #ind (x > 0) $
#let ind = $bb(1)$

// Author names / points: #authorpoints("Sadegh, W5 Monday", 8)
#let authorpoints(who, points) = text(size: 14.4pt)[
  #raw("(" + str(points) + " points)") #raw("[" + who + "]")
]

// ---------- Code style (lstlisting "mystyle") ----------
#show raw.where(block: true): it => block(
  width: 100%,
  fill: rgb("#F7F7F7"),
  stroke: 0.5pt + rgb("#EEEEEE"),
  inset: 8pt,
  radius: 0pt,
  text(size: 10pt, it),
)
#set raw(tab-size: 4, lang: "python")

// ---------- Title ----------
#align(center)[
  #v(1em)
  #text(size: 17pt, weight: "bold")[Machine Learning A (2026) \ Home Assignment 5]
  #v(1em)
  #text(size: 14pt, fill: red)[Ludovico Maria Spitaleri, DMH249]
  #v(1em)
]

// Please leave the table of contents as is, for the ease of navigation for TAs
#outline()
// #pagebreak() // Start a new page after the table of contents


= On the Role of Dependence #authorpoints("Sadegh, W5 Monday", 8)


= On Confidence Intervals #authorpoints("Sadegh, W5 Monday", 12)

== Part 1

== Part 2

= Loss Range Correction in Generalization Bounds #authorpoints("Sadegh, W5 Monday", 10)

== (i)

== (ii)


= Convolutional Neural Networks #authorpoints("Christian, W5 Friday", 70)

== Sobel filter #authorpoints("Christian, W5 Friday", 24)

The Sobel filter is implemented with a single `nn.Conv2d` with one input and two output channels. The magnitude $G$ is obtained by squaring, summing over the channel axis and taking the square root.

```python
Kx = torch.tensor([[1., 0., -1.],
                   [2., 0., -2.],
                   [1., 0., -1.]])
Ky = torch.tensor([[ 1.,  2.,  1.],
                   [ 0.,  0.,  0.],
                   [-1., -2., -1.]])
K = torch.stack([Kx, Ky]).unsqueeze(1)
K = torch.flip(K, dims=[-2, -1])

conv = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
conv.weight = nn.Parameter(K, requires_grad=False)

g = conv(x)
G = torch.sqrt(torch.sum(torch.square(g), dim=1))
Gx, Gy = g[0, 0], g[0, 1]
```

#figure(
  image("src/figures/sobel.png", width: 100%),
  caption: [The Sobel feature maps $G_x$, $G_y$ and $G$.],
)

`nn.Conv2d` does not flip, it computes the cross-correlation $[K star I]_(i j) = sum_(u,v) K_(u v) I_(i+u, j+v)$. Using $G_x, G_y$ unchanged would therefore compute the correlation with those matrices, which equals the convolution with the _flipped_ matrices, so $G_x$ and $G_y$ would come out with the opposite sign. To implement the convolution the kernels must be flipped before assigning them as weights.

The final map $G$ is unaffected by this choice, since it only depends on $G_x^2$ and $G_y^2$ and a flip of these antisymmetric kernels only changes the sign of $G_x$ and $G_y$.

== Convolutional neural networks #authorpoints("Christian, W5 Friday", 24)

```python
class Net(nn.Module):
    def __init__(
        self,
        img_size=28,
        in_channels=3,  # r, g, b
        n_maps=64,
        kernel_size=5,
        pool_size=2,
        n_classes=43,
    ):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, n_maps, kernel_size=kernel_size)
        size = img_size - kernel_size + 1
        self.pool1 = nn.MaxPool2d(pool_size)
        size = size // pool_size
        self.conv2 = nn.Conv2d(n_maps, n_maps, kernel_size=kernel_size)
        size = size - kernel_size + 1
        self.pool2 = nn.MaxPool2d(pool_size)
        size = size // pool_size

        self.fc2 = nn.Linear(n_maps * size * size, n_classes)

    def forward(self, x):
        x = self.pool1(F.elu(self.conv1(x)))
        x = self.pool2(F.elu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.fc2(x)
```

Printing the model gives the expected output:
```
Net(
  (conv1): Conv2d(3, 64, kernel_size=(5, 5), stride=(1, 1))
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(64, 64, kernel_size=(5, 5), stride=(1, 1))
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (fc2): Linear(in_features=1024, out_features=43, bias=True)
)
```

== Augmentation #authorpoints("Christian, W5 Friday", 22)

During training, each image is
+ resized to $32 times 32$
+ randomly rotated by an angle between $[-5°, 5°]$
+ randomly cropped to $28 times 28$
+ colour-jittered with a brightness factor between $[0.2, 1.8]$ and contrast factor between $[0.6, 1.4]$
+ if it's part of the classes $11, 12, 13, 17, 18, 26, 30, 35$, the image is horizontally flipped with probability 0.5.

If not training, all images are just cropped to $28 times 28$ in the center to have a consistent size with the training images.

Since augmentation must not change the label, only signs of classes 11, 12, 13, 17, 18, 26, 30 and 35 are horizontally flipped. This is because they are almost symmetric, so a flipped image still shows the same class. For other classes a flip changes the meaning: a flipped arrow becomes the opposite direction, or a speed limit could have a wrong number on it after the transformation.

An extra useful transformations that could be added in the training is a change in the prespective:

```python
image = transforms.RandomPerspective(distortion_scale=0.15, p=0.5)(image)
```

This would help the model train on recognizing signs even when not show on a perfect angle. This is a very common situation in real-world driving when encountering signs, especially if on the side of tight curves.


// ---------- Placeholder figure with two subfigures ----------
// // Replace the rects with image("your-file.png", width: 100%)
// #let placeholder = rect(width: 100%, height: 4cm, fill: luma(230))

// #grid(
//   columns: (0.45fr, 0.45fr),
//   column-gutter: 1fr,
//   [
//     #figure(
//       placeholder,
//       caption: [Lorem Ipsum A],
//       kind: "subfigure",
//       supplement: none,
//       numbering: "(a)",
//     ) <fig:subfig_a>
//   ],
//   [
//     #figure(
//       placeholder,
//       caption: [Lorem Ipsum B],
//       kind: "subfigure",
//       supplement: none,
//       numbering: "(a)",
//     ) <fig:subfig_b>
//   ],
// )


// ---------- Placeholder figure ----------
// #figure(
//   rect(width: 50%, height: 4cm, fill: luma(230)),
//   // Dummy caption generated with lorem
//   caption: [#lorem(15)],
// ) <fig:placeholder>


// ---------- Placeholder code ----------
// ```python
// # Example code
// # Creating an example array
// data = np.array([5, 2, 8, 1, 6])

// # Calculating cumulative sum using cumsum
// cumulative_sum = np.cumsum(data)
// ```


// ---------- Bibliography ----------
// If you have references, put a bibliography.bib (or .yml) next to this file
// and uncomment:
// #bibliography("bibliography.bib")
