/**
 *  @file   OrthogonalPolynomialSlidingWindow.cpp
 *  @brief  OrthogonalPolynomialSlidingWindow class implementation
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
#include "OrthogonalPolynomialSlidingWindow.hpp"

#include <cmath>

#include <algorithm>

OrthogonalPolynomialSlidingWindow::OrthogonalPolynomialSlidingWindow(int degree, int windowSize)
{
  if (degree < 0)
      throw ("Degree must not be negative.");

  if (degree >= windowSize)
      throw ("Window size must be greater than degree.");

  this->degree = degree;
  this->windowSize = windowSize;

  init();

  double m = windowSize - 1;

  std::vector<double> delta;
  std::vector<double> qnorm;

  for(int i=0;i<degree +2;++i)
  {
    delta.push_back(double());
    qnorm.push_back(double());
  }

  delta[0] = m + 1;
  qnorm[0] = delta[0];

  double qmpo = (m + 1) * (m + 1);
  for (int i = 1; i <= degree + 1; i++)
  {
    double qi = i * i;
    delta[i] = qi * (qmpo - qi) / (16 * qi - 4);
    qnorm[i] = qnorm[i - 1] * delta[i];
  }

  for(int i=0;i<degree +1;++i)
  {
    this->delta.push_back(delta[i]);
    this->qnorm.push_back(qnorm[i]);
  }

  std::vector<std::vector<double> > gamma;

  for(int i=0;i<degree +1;++i)
    gamma.push_back(std::vector<double>());

  std::vector<double> initvec;
  initvec.push_back(1);
  initvec.push_back(0);
  initvec.push_back(0);

  gamma[0] = initvec;
  initvec.clear();

  if (degree > 0)
  {

    initvec.push_back(1);
    initvec.push_back(1);
    initvec.push_back(0);
    initvec.push_back(0);

    gamma[1] = initvec;

    for(int i=2;i<gamma.size();++i)
    {
      for(int j=0;j<=i+3;++j)
      {
        gamma[i].push_back(double());
      }

      for(int j=0;j<=i;++j)
      {
        gamma[i][j] = gamma[i - 1][j + 1] * delta[j + 1]
                          + gamma[i - 1][j] - gamma[i - 2][j] * delta[i - 1];

        if (j > 0)
          gamma[i][j] += gamma[i - 1][j - 1];
      }
    }
  }

  for(int i=0;i<degree +1;++i)
  {
    updateCoefs.push_back(std::vector<double>());
  }

  for(int i=0;i<2;++i)
  {
    updateCoefs[0].push_back(double());
  }

  updateCoefs[0][0] = 1;
  updateCoefs[0][1] = -1;

  if (degree > 0)
  {
    for (int i = 1; i < updateCoefs.size(); ++i)
    {
      for(int j=0;j<i+2;++j)
      {
        updateCoefs[i].push_back(double());
      }
      updateCoefs[i][0] = updateCoefs[i - 1][0] * i / (4.0 * i - 2)
              * (m + i + 1);
      updateCoefs[i][1] = updateCoefs[i - 1][1] * -i / (4.0 * i - 2)
              * (m - i + 1);

      for (int j = 2; j < i + 2; j++)
        updateCoefs[i][j] = -gamma[i][j - 2];
     }
  }
}

void OrthogonalPolynomialSlidingWindow::init()
{
  this->alpha.clear();
  this->values.clear();

  for(int i=0;i<degree +1;++i)
    this->alpha.push_back(double());

  for(int i=0;i<windowSize;++i)
    this->values.push_back(double());

  this->valueIdx = 0;
  this->sqsum = 0;
}

void OrthogonalPolynomialSlidingWindow::restart()
{
  std::vector<double> valuesOld = values;
  int valueIdxOld = valueIdx;

  init();

  for (int i = 0; i < windowSize; i++)
    update(valuesOld[(valueIdxOld + i) % valuesOld.size()]);
}

void OrthogonalPolynomialSlidingWindow::update(double y)
{
  for (int i = 0; i <= degree; i++)
  {
    double alphaNew = alpha[i] + y * updateCoefs[i][0]
            + values[valueIdx] * updateCoefs[i][1];
    for (int j = 0; j < i; j++)
        alphaNew += updateCoefs[i][j + 2] * alpha[j];
    alpha[i] = alphaNew;
  }
  sqsum += y * y - values[valueIdx] * values[valueIdx];
  values[valueIdx++] = y;
  if (valueIdx >= values.size())
    valueIdx = 0;
}

double OrthogonalPolynomialSlidingWindow::eval(double x)
{
  std::vector<double> q;
  for(int i=0;i<degree +1;++i)
    q.push_back(double());

  q[degree] = alpha[degree] / qnorm[degree];

  if (degree > 0)
  {
    x = x - (windowSize - 1) / 2.0;
    q[degree - 1] = alpha[degree - 1] / qnorm[degree - 1] + x * q[degree];

    for (int i = degree - 2; i >= 0; i--)
      q[i] = alpha[i] / qnorm[i] + x * q[i + 1] - delta[i + 1] * q[i + 2];
  }
  return q[0];
}

std::vector<double> OrthogonalPolynomialSlidingWindow::getOrthogonalCoefficients()
{
  std::vector<double> coefs;
  for(int i=0;i<degree +1;++i)
    coefs.push_back(double());
  for (int i = 0; i < coefs.size(); i++)
      coefs[i] = alpha[i] / qnorm[i];
  return coefs;
}

double OrthogonalPolynomialSlidingWindow::getApproximationError()
{
  double tmpSum = sqsum;
  for (int i = 0; i <= degree; i++)
    tmpSum -= alpha[i] * alpha[i] / qnorm[i];

  return std::max(0.0, tmpSum);
}

int OrthogonalPolynomialSlidingWindow::getExtremumPosition()
{
  if (degree < 2)
      throw "Unable to calculate the extremum of the current solution (degree must be >= 2)";
  if (alpha[2] == 0)
      throw "Unable to calculate the extremum of the current solution (curve is zero)";
  return (int) std::floor(((windowSize - 1) / 2.0 - (alpha[1] * qnorm[2])
          / (2.0 * alpha[2] * qnorm[1])) +0.5
  );
}
