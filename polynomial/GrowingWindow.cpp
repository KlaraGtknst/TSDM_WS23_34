/**
 *  @file   GrowingWindow.cpp
 *  @brief  GrowingWindow class implementation
 *  @date   26.03.2013
 *  @author Dr. Thiemo Gruber, Peter Rudolph
 *
 *  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES\n
 *  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF\n
 *  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR\n
 *  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES\n
 *  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN\n
 *  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF\n
 *  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.\n
 *
 *  @copyright Intelligent Embedded Systems, University of Kassel, 2013. All rights reserved.
 *  @license This library is released under <a href="http://www.gnu.org/licenses/lgpl-3.0.de.html">LGPLv3</a>.
 */
#include "GrowingWindow.hpp"

#include <cmath>
#include <algorithm>
#include <iostream>

/**
 *  @brief Round macro
 *  @param x Value to round
 *  @return Rounded x
 */
#define round(x) (floor((x) + 0.5))

/**
 *  @brief Sign function
 *  @param x Value to get sign from
 *  @return Sign of x
 */
int signum(double x)
{
  if(x>=0)
    return 1;
  else
    return -1;
}


GrowingWindow::GrowingWindow(unsigned int degree)
{
  alpha_ = std::vector<double>(degree + 2);
  beta_ = std::vector<double>(degree + 1);
  d_ = std::vector<double>(degree + 2);
  degree_ = degree;
  sigma_ = 0;
  n_ = 0;
  nb_pts_ = 0;
  ysqsum_ = 0;
}

GrowingWindow::~GrowingWindow()
{

}

void GrowingWindow::reset()
{
  n_ = 0;
  nb_pts_ = 0;
}

void GrowingWindow::update(double y)
{
  if (n_ == 0)
  {
    sigma_ = 1;
    alpha_[0] = 0;
    d_[0] = y;
    n_ = 1;
    nb_pts_ = 1;
    ysqsum_ = y * y;

  }
  else
  {
    int nNew = n_ + (n_ < degree_ + 2 ? 1 : 0);
    double sigmaNew = sqrt(sigma_ * sigma_ + 1);
    double c = sigma_ / sigmaNew;
    double s = 1 / sigmaNew;
    sigma_ = sigmaNew;
    double theta1 = c * s * (nb_pts_ - alpha_[0]);
    double theta2 = degree_ > 0 ? -s * beta_[0] : 0;
    alpha_[nNew - 1] = c * c * nb_pts_ + s * s * alpha_[0];
    alpha_[0] = c * c * alpha_[0] + s * s * nb_pts_;
    beta_[0] = c * beta_[0];
    d_[nNew - 1] = c * y - s * d_[0];
    d_[0] = c * d_[0] + s * y;

    for (int i = 1; i < nNew - 1; i++)
    {
      if (beta_[i - 1] * beta_[i - 1] + theta1 * theta1 == 0)
      {
        n_ = i;
        return;
      }

      double zeta = sqrt(beta_[i - 1] * beta_[i - 1] + theta1 * theta1);
      c = beta_[i - 1] / zeta;
      s = theta1 / zeta;
      beta_[i - 1] = zeta;
      double t = c * c * alpha_[i] + 2 * c * s * theta2
          + s * s * alpha_[nNew - 1];
      double r = c * c * alpha_[nNew - 1] - 2 * c * s * theta2
          + s * s * alpha_[i];
      theta1 = c * s * (alpha_[nNew - 1] - alpha_[i])
          + theta2 * (c * c - s * s);
      theta2 = -s * beta_[i];
      alpha_[i] = t;
      alpha_[nNew - 1] = r;
      beta_[i] = c * beta_[i];
      t = d_[i];
      d_[i] = c * t + s * d_[nNew - 1];
      d_[nNew - 1] = c * d_[nNew - 1] - s * t;
    }

    if (theta1 != 0)
    {
      beta_[nNew - 2] = fabs(theta1);
      d_[nNew - 1] = signum(theta1) * d_[nNew - 1];
      n_ = nNew;
    }
    else
    {
      std::cout << "Warning: theta1 = 0" << std::endl;
    }

    nb_pts_++;
    ysqsum_ += y * y;

  }
}

double GrowingWindow::eval(double x)
{
  if (n_ <= degree_)
    throw "Unable to evaluate polynomial for less than degree + 1 points.";

  std::vector<double> y(d_.size());

  if (degree_ == 0)
  {
    y[0] = d_[0];

  }
  else if (degree_ == 1)
  {
    y[0] = d_[0] + d_[1] / beta_[0] * (x - alpha_[0]);

  }
  else
  {
    y[degree_] = d_[degree_] / beta_[degree_ - 1];
    y[degree_ - 1] = (d_[degree_ - 1] + y[degree_] * (x - alpha_[degree_ - 1]))
        / beta_[degree_ - 2];

    for (int i = degree_ - 2; i >= 1; i--)
    {
      y[i] = (d_[i] + y[i + 1] * (x - alpha_[i]) - y[i + 2] * beta_[i])
          / beta_[i - 1];
    }

    y[0] = d_[0] + y[1] * (x - alpha_[0]) - y[2] * beta_[0];
  }

  return y[0] / sigma_;
}

int GrowingWindow::getNumSamples()
{
  return nb_pts_;
}

std::vector<double> GrowingWindow::getOrthogonalCoefficients()
{
  if (n_ <= degree_)
    throw "Unable to evaluate polynomial for less than degree + 1 points.";

  std::vector<double> c(degree_ + 1);
  double nf = 1 / sigma_;

  c[0] = nf * d_[0];

  for (int i = 1; i <= degree_; i++)
  {
    nf /= beta_[i - 1];
    c[i] = d_[i] * nf;
  }

  return c;
}

double GrowingWindow::getApproximationError()
{
  if (n_ <= degree_)
    throw "Unable to evaluate polynomial for less than degree + 1 points.";

  double dsqsum = 0;

  for (int i = 0; i <= degree_; i++)
    dsqsum += d_[i] * d_[i];

  if (ysqsum_ - dsqsum < -1e-8)
    std::cout << "Warning: Approximation error negative!" << std::endl;

  return std::max(0.0, ysqsum_ - dsqsum);
}

int GrowingWindow::getExtremumPosition()
{
  if (degree_ < 2 || n_ <= degree_)
      throw "Unable to calculate the extremum of the current solution (degree must be >= 2, solution size must be > degree)";

  if (d_[2] == 0)
      throw "Unable to calculate the extremum of the current solution (curve is zero)";

  return (int) round((alpha_[0] + alpha_[1]) / 2 - d_[1] * beta_[1] / (2 * d_[2]));
}
