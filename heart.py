{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "R_YGtYHtPf-J"
   },
   "outputs": [],
   "source": [
    "#to classifies a person as having a cadio vascular disease or not"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "bnK--G0oQB8K"
   },
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import pickle\n",
    "from matplotlib import rcParams\n",
    "from matplotlib.cm import rainbow\n",
    "%matplotlib inline\n",
    "import warnings\n",
    "import io\n",
    "import seaborn as sns \n",
    "\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "ylJvXPWxbwut"
   },
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 74,
     "resources": {
      "http://localhost:8080/nbextensions/google.colab/files.js": {
       "data": "Ly8gQ29weXJpZ2h0IDIwMTcgR29vZ2xlIExMQwovLwovLyBMaWNlbnNlZCB1bmRlciB0aGUgQXBhY2hlIExpY2Vuc2UsIFZlcnNpb24gMi4wICh0aGUgIkxpY2Vuc2UiKTsKLy8geW91IG1heSBub3QgdXNlIHRoaXMgZmlsZSBleGNlcHQgaW4gY29tcGxpYW5jZSB3aXRoIHRoZSBMaWNlbnNlLgovLyBZb3UgbWF5IG9idGFpbiBhIGNvcHkgb2YgdGhlIExpY2Vuc2UgYXQKLy8KLy8gICAgICBodHRwOi8vd3d3LmFwYWNoZS5vcmcvbGljZW5zZXMvTElDRU5TRS0yLjAKLy8KLy8gVW5sZXNzIHJlcXVpcmVkIGJ5IGFwcGxpY2FibGUgbGF3IG9yIGFncmVlZCB0byBpbiB3cml0aW5nLCBzb2Z0d2FyZQovLyBkaXN0cmlidXRlZCB1bmRlciB0aGUgTGljZW5zZSBpcyBkaXN0cmlidXRlZCBvbiBhbiAiQVMgSVMiIEJBU0lTLAovLyBXSVRIT1VUIFdBUlJBTlRJRVMgT1IgQ09ORElUSU9OUyBPRiBBTlkgS0lORCwgZWl0aGVyIGV4cHJlc3Mgb3IgaW1wbGllZC4KLy8gU2VlIHRoZSBMaWNlbnNlIGZvciB0aGUgc3BlY2lmaWMgbGFuZ3VhZ2UgZ292ZXJuaW5nIHBlcm1pc3Npb25zIGFuZAovLyBsaW1pdGF0aW9ucyB1bmRlciB0aGUgTGljZW5zZS4KCi8qKgogKiBAZmlsZW92ZXJ2aWV3IEhlbHBlcnMgZm9yIGdvb2dsZS5jb2xhYiBQeXRob24gbW9kdWxlLgogKi8KKGZ1bmN0aW9uKHNjb3BlKSB7CmZ1bmN0aW9uIHNwYW4odGV4dCwgc3R5bGVBdHRyaWJ1dGVzID0ge30pIHsKICBjb25zdCBlbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnc3BhbicpOwogIGVsZW1lbnQudGV4dENvbnRlbnQgPSB0ZXh0OwogIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKHN0eWxlQXR0cmlidXRlcykpIHsKICAgIGVsZW1lbnQuc3R5bGVba2V5XSA9IHN0eWxlQXR0cmlidXRlc1trZXldOwogIH0KICByZXR1cm4gZWxlbWVudDsKfQoKLy8gTWF4IG51bWJlciBvZiBieXRlcyB3aGljaCB3aWxsIGJlIHVwbG9hZGVkIGF0IGEgdGltZS4KY29uc3QgTUFYX1BBWUxPQURfU0laRSA9IDEwMCAqIDEwMjQ7Ci8vIE1heCBhbW91bnQgb2YgdGltZSB0byBibG9jayB3YWl0aW5nIGZvciB0aGUgdXNlci4KY29uc3QgRklMRV9DSEFOR0VfVElNRU9VVF9NUyA9IDMwICogMTAwMDsKCmZ1bmN0aW9uIF91cGxvYWRGaWxlcyhpbnB1dElkLCBvdXRwdXRJZCkgewogIGNvbnN0IHN0ZXBzID0gdXBsb2FkRmlsZXNTdGVwKGlucHV0SWQsIG91dHB1dElkKTsKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIC8vIENhY2hlIHN0ZXBzIG9uIHRoZSBvdXRwdXRFbGVtZW50IHRvIG1ha2UgaXQgYXZhaWxhYmxlIGZvciB0aGUgbmV4dCBjYWxsCiAgLy8gdG8gdXBsb2FkRmlsZXNDb250aW51ZSBmcm9tIFB5dGhvbi4KICBvdXRwdXRFbGVtZW50LnN0ZXBzID0gc3RlcHM7CgogIHJldHVybiBfdXBsb2FkRmlsZXNDb250aW51ZShvdXRwdXRJZCk7Cn0KCi8vIFRoaXMgaXMgcm91Z2hseSBhbiBhc3luYyBnZW5lcmF0b3IgKG5vdCBzdXBwb3J0ZWQgaW4gdGhlIGJyb3dzZXIgeWV0KSwKLy8gd2hlcmUgdGhlcmUgYXJlIG11bHRpcGxlIGFzeW5jaHJvbm91cyBzdGVwcyBhbmQgdGhlIFB5dGhvbiBzaWRlIGlzIGdvaW5nCi8vIHRvIHBvbGwgZm9yIGNvbXBsZXRpb24gb2YgZWFjaCBzdGVwLgovLyBUaGlzIHVzZXMgYSBQcm9taXNlIHRvIGJsb2NrIHRoZSBweXRob24gc2lkZSBvbiBjb21wbGV0aW9uIG9mIGVhY2ggc3RlcCwKLy8gdGhlbiBwYXNzZXMgdGhlIHJlc3VsdCBvZiB0aGUgcHJldmlvdXMgc3RlcCBhcyB0aGUgaW5wdXQgdG8gdGhlIG5leHQgc3RlcC4KZnVuY3Rpb24gX3VwbG9hZEZpbGVzQ29udGludWUob3V0cHV0SWQpIHsKICBjb25zdCBvdXRwdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQob3V0cHV0SWQpOwogIGNvbnN0IHN0ZXBzID0gb3V0cHV0RWxlbWVudC5zdGVwczsKCiAgY29uc3QgbmV4dCA9IHN0ZXBzLm5leHQob3V0cHV0RWxlbWVudC5sYXN0UHJvbWlzZVZhbHVlKTsKICByZXR1cm4gUHJvbWlzZS5yZXNvbHZlKG5leHQudmFsdWUucHJvbWlzZSkudGhlbigodmFsdWUpID0+IHsKICAgIC8vIENhY2hlIHRoZSBsYXN0IHByb21pc2UgdmFsdWUgdG8gbWFrZSBpdCBhdmFpbGFibGUgdG8gdGhlIG5leHQKICAgIC8vIHN0ZXAgb2YgdGhlIGdlbmVyYXRvci4KICAgIG91dHB1dEVsZW1lbnQubGFzdFByb21pc2VWYWx1ZSA9IHZhbHVlOwogICAgcmV0dXJuIG5leHQudmFsdWUucmVzcG9uc2U7CiAgfSk7Cn0KCi8qKgogKiBHZW5lcmF0b3IgZnVuY3Rpb24gd2hpY2ggaXMgY2FsbGVkIGJldHdlZW4gZWFjaCBhc3luYyBzdGVwIG9mIHRoZSB1cGxvYWQKICogcHJvY2Vzcy4KICogQHBhcmFtIHtzdHJpbmd9IGlucHV0SWQgRWxlbWVudCBJRCBvZiB0aGUgaW5wdXQgZmlsZSBwaWNrZXIgZWxlbWVudC4KICogQHBhcmFtIHtzdHJpbmd9IG91dHB1dElkIEVsZW1lbnQgSUQgb2YgdGhlIG91dHB1dCBkaXNwbGF5LgogKiBAcmV0dXJuIHshSXRlcmFibGU8IU9iamVjdD59IEl0ZXJhYmxlIG9mIG5leHQgc3RlcHMuCiAqLwpmdW5jdGlvbiogdXBsb2FkRmlsZXNTdGVwKGlucHV0SWQsIG91dHB1dElkKSB7CiAgY29uc3QgaW5wdXRFbGVtZW50ID0gZG9jdW1lbnQuZ2V0RWxlbWVudEJ5SWQoaW5wdXRJZCk7CiAgaW5wdXRFbGVtZW50LmRpc2FibGVkID0gZmFsc2U7CgogIGNvbnN0IG91dHB1dEVsZW1lbnQgPSBkb2N1bWVudC5nZXRFbGVtZW50QnlJZChvdXRwdXRJZCk7CiAgb3V0cHV0RWxlbWVudC5pbm5lckhUTUwgPSAnJzsKCiAgY29uc3QgcGlja2VkUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICBpbnB1dEVsZW1lbnQuYWRkRXZlbnRMaXN0ZW5lcignY2hhbmdlJywgKGUpID0+IHsKICAgICAgcmVzb2x2ZShlLnRhcmdldC5maWxlcyk7CiAgICB9KTsKICB9KTsKCiAgY29uc3QgY2FuY2VsID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnYnV0dG9uJyk7CiAgaW5wdXRFbGVtZW50LnBhcmVudEVsZW1lbnQuYXBwZW5kQ2hpbGQoY2FuY2VsKTsKICBjYW5jZWwudGV4dENvbnRlbnQgPSAnQ2FuY2VsIHVwbG9hZCc7CiAgY29uc3QgY2FuY2VsUHJvbWlzZSA9IG5ldyBQcm9taXNlKChyZXNvbHZlKSA9PiB7CiAgICBjYW5jZWwub25jbGljayA9ICgpID0+IHsKICAgICAgcmVzb2x2ZShudWxsKTsKICAgIH07CiAgfSk7CgogIC8vIENhbmNlbCB1cGxvYWQgaWYgdXNlciBoYXNuJ3QgcGlja2VkIGFueXRoaW5nIGluIHRpbWVvdXQuCiAgY29uc3QgdGltZW91dFByb21pc2UgPSBuZXcgUHJvbWlzZSgocmVzb2x2ZSkgPT4gewogICAgc2V0VGltZW91dCgoKSA9PiB7CiAgICAgIHJlc29sdmUobnVsbCk7CiAgICB9LCBGSUxFX0NIQU5HRV9USU1FT1VUX01TKTsKICB9KTsKCiAgLy8gV2FpdCBmb3IgdGhlIHVzZXIgdG8gcGljayB0aGUgZmlsZXMuCiAgY29uc3QgZmlsZXMgPSB5aWVsZCB7CiAgICBwcm9taXNlOiBQcm9taXNlLnJhY2UoW3BpY2tlZFByb21pc2UsIHRpbWVvdXRQcm9taXNlLCBjYW5jZWxQcm9taXNlXSksCiAgICByZXNwb25zZTogewogICAgICBhY3Rpb246ICdzdGFydGluZycsCiAgICB9CiAgfTsKCiAgaWYgKCFmaWxlcykgewogICAgcmV0dXJuIHsKICAgICAgcmVzcG9uc2U6IHsKICAgICAgICBhY3Rpb246ICdjb21wbGV0ZScsCiAgICAgIH0KICAgIH07CiAgfQoKICBjYW5jZWwucmVtb3ZlKCk7CgogIC8vIERpc2FibGUgdGhlIGlucHV0IGVsZW1lbnQgc2luY2UgZnVydGhlciBwaWNrcyBhcmUgbm90IGFsbG93ZWQuCiAgaW5wdXRFbGVtZW50LmRpc2FibGVkID0gdHJ1ZTsKCiAgZm9yIChjb25zdCBmaWxlIG9mIGZpbGVzKSB7CiAgICBjb25zdCBsaSA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2xpJyk7CiAgICBsaS5hcHBlbmQoc3BhbihmaWxlLm5hbWUsIHtmb250V2VpZ2h0OiAnYm9sZCd9KSk7CiAgICBsaS5hcHBlbmQoc3BhbigKICAgICAgICBgKCR7ZmlsZS50eXBlIHx8ICduL2EnfSkgLSAke2ZpbGUuc2l6ZX0gYnl0ZXMsIGAgKwogICAgICAgIGBsYXN0IG1vZGlmaWVkOiAkewogICAgICAgICAgICBmaWxlLmxhc3RNb2RpZmllZERhdGUgPyBmaWxlLmxhc3RNb2RpZmllZERhdGUudG9Mb2NhbGVEYXRlU3RyaW5nKCkgOgogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAnbi9hJ30gLSBgKSk7CiAgICBjb25zdCBwZXJjZW50ID0gc3BhbignMCUgZG9uZScpOwogICAgbGkuYXBwZW5kQ2hpbGQocGVyY2VudCk7CgogICAgb3V0cHV0RWxlbWVudC5hcHBlbmRDaGlsZChsaSk7CgogICAgY29uc3QgZmlsZURhdGFQcm9taXNlID0gbmV3IFByb21pc2UoKHJlc29sdmUpID0+IHsKICAgICAgY29uc3QgcmVhZGVyID0gbmV3IEZpbGVSZWFkZXIoKTsKICAgICAgcmVhZGVyLm9ubG9hZCA9IChlKSA9PiB7CiAgICAgICAgcmVzb2x2ZShlLnRhcmdldC5yZXN1bHQpOwogICAgICB9OwogICAgICByZWFkZXIucmVhZEFzQXJyYXlCdWZmZXIoZmlsZSk7CiAgICB9KTsKICAgIC8vIFdhaXQgZm9yIHRoZSBkYXRhIHRvIGJlIHJlYWR5LgogICAgbGV0IGZpbGVEYXRhID0geWllbGQgewogICAgICBwcm9taXNlOiBmaWxlRGF0YVByb21pc2UsCiAgICAgIHJlc3BvbnNlOiB7CiAgICAgICAgYWN0aW9uOiAnY29udGludWUnLAogICAgICB9CiAgICB9OwoKICAgIC8vIFVzZSBhIGNodW5rZWQgc2VuZGluZyB0byBhdm9pZCBtZXNzYWdlIHNpemUgbGltaXRzLiBTZWUgYi82MjExNTY2MC4KICAgIGxldCBwb3NpdGlvbiA9IDA7CiAgICB3aGlsZSAocG9zaXRpb24gPCBmaWxlRGF0YS5ieXRlTGVuZ3RoKSB7CiAgICAgIGNvbnN0IGxlbmd0aCA9IE1hdGgubWluKGZpbGVEYXRhLmJ5dGVMZW5ndGggLSBwb3NpdGlvbiwgTUFYX1BBWUxPQURfU0laRSk7CiAgICAgIGNvbnN0IGNodW5rID0gbmV3IFVpbnQ4QXJyYXkoZmlsZURhdGEsIHBvc2l0aW9uLCBsZW5ndGgpOwogICAgICBwb3NpdGlvbiArPSBsZW5ndGg7CgogICAgICBjb25zdCBiYXNlNjQgPSBidG9hKFN0cmluZy5mcm9tQ2hhckNvZGUuYXBwbHkobnVsbCwgY2h1bmspKTsKICAgICAgeWllbGQgewogICAgICAgIHJlc3BvbnNlOiB7CiAgICAgICAgICBhY3Rpb246ICdhcHBlbmQnLAogICAgICAgICAgZmlsZTogZmlsZS5uYW1lLAogICAgICAgICAgZGF0YTogYmFzZTY0LAogICAgICAgIH0sCiAgICAgIH07CiAgICAgIHBlcmNlbnQudGV4dENvbnRlbnQgPQogICAgICAgICAgYCR7TWF0aC5yb3VuZCgocG9zaXRpb24gLyBmaWxlRGF0YS5ieXRlTGVuZ3RoKSAqIDEwMCl9JSBkb25lYDsKICAgIH0KICB9CgogIC8vIEFsbCBkb25lLgogIHlpZWxkIHsKICAgIHJlc3BvbnNlOiB7CiAgICAgIGFjdGlvbjogJ2NvbXBsZXRlJywKICAgIH0KICB9Owp9CgpzY29wZS5nb29nbGUgPSBzY29wZS5nb29nbGUgfHwge307CnNjb3BlLmdvb2dsZS5jb2xhYiA9IHNjb3BlLmdvb2dsZS5jb2xhYiB8fCB7fTsKc2NvcGUuZ29vZ2xlLmNvbGFiLl9maWxlcyA9IHsKICBfdXBsb2FkRmlsZXMsCiAgX3VwbG9hZEZpbGVzQ29udGludWUsCn07Cn0pKHNlbGYpOwo=",
       "headers": [
        [
         "content-type",
         "application/javascript"
        ]
       ],
       "ok": true,
       "status": 200,
       "status_text": ""
      }
     }
    },
    "colab_type": "code",
    "id": "n74TAaYk-DyB",
    "outputId": "0276162d-7bb9-45f3-d5cd-189699064755"
   },
   "outputs": [],
   "source": [
    " "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 195
    },
    "colab_type": "code",
    "id": "j3V_t0QjE28c",
    "outputId": "f7b48832-c01b-41e0-abb0-3b6f1fa1a2d3"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>145</td>\n",
       "      <td>233</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>150</td>\n",
       "      <td>0</td>\n",
       "      <td>2.3</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>37</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>130</td>\n",
       "      <td>250</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>187</td>\n",
       "      <td>0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>41</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>130</td>\n",
       "      <td>204</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>172</td>\n",
       "      <td>0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>56</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>120</td>\n",
       "      <td>236</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>178</td>\n",
       "      <td>0</td>\n",
       "      <td>0.8</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>57</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>120</td>\n",
       "      <td>354</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>163</td>\n",
       "      <td>1</td>\n",
       "      <td>0.6</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  \\\n",
       "0   63    1   3       145   233    1        0      150      0      2.3      0   \n",
       "1   37    1   2       130   250    0        1      187      0      3.5      0   \n",
       "2   41    0   1       130   204    0        0      172      0      1.4      2   \n",
       "3   56    1   1       120   236    0        1      178      0      0.8      2   \n",
       "4   57    0   0       120   354    0        1      163      1      0.6      2   \n",
       "\n",
       "   ca  thal  target  \n",
       "0   0     1       1  \n",
       "1   0     2       1  \n",
       "2   0     2       1  \n",
       "3   0     2       1  \n",
       "4   0     2       1  "
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df = pd.read_csv('heart_disease_dataset.csv')\n",
    "df.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 402
    },
    "colab_type": "code",
    "id": "mIGiFPExSoNI",
    "outputId": "6657c757-d42b-49e0-8e9e-71a64a2fc083"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>298</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>299</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>300</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>301</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>302</th>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>303 rows × 14 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       age    sex     cp  trestbps   chol    fbs  restecg  thalach  exang  \\\n",
       "0    False  False  False     False  False  False    False    False  False   \n",
       "1    False  False  False     False  False  False    False    False  False   \n",
       "2    False  False  False     False  False  False    False    False  False   \n",
       "3    False  False  False     False  False  False    False    False  False   \n",
       "4    False  False  False     False  False  False    False    False  False   \n",
       "..     ...    ...    ...       ...    ...    ...      ...      ...    ...   \n",
       "298  False  False  False     False  False  False    False    False  False   \n",
       "299  False  False  False     False  False  False    False    False  False   \n",
       "300  False  False  False     False  False  False    False    False  False   \n",
       "301  False  False  False     False  False  False    False    False  False   \n",
       "302  False  False  False     False  False  False    False    False  False   \n",
       "\n",
       "     oldpeak  slope     ca   thal  target  \n",
       "0      False  False  False  False   False  \n",
       "1      False  False  False  False   False  \n",
       "2      False  False  False  False   False  \n",
       "3      False  False  False  False   False  \n",
       "4      False  False  False  False   False  \n",
       "..       ...    ...    ...    ...     ...  \n",
       "298    False  False  False  False   False  \n",
       "299    False  False  False  False   False  \n",
       "300    False  False  False  False   False  \n",
       "301    False  False  False  False   False  \n",
       "302    False  False  False  False   False  \n",
       "\n",
       "[303 rows x 14 columns]"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 269
    },
    "colab_type": "code",
    "id": "ChdiYjpwTQQk",
    "outputId": "13950aae-a7d3-448d-9aae-ec86e544c504"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "age         0\n",
       "sex         0\n",
       "cp          0\n",
       "trestbps    0\n",
       "chol        0\n",
       "fbs         0\n",
       "restecg     0\n",
       "thalach     0\n",
       "exang       0\n",
       "oldpeak     0\n",
       "slope       0\n",
       "ca          0\n",
       "thal        0\n",
       "target      0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()# to check whether our data set contain null values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 336
    },
    "colab_type": "code",
    "id": "ehBgITDmsp3O",
    "outputId": "a490731b-ddb0-4cd5-e566-0a740c1da8f0"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 303 entries, 0 to 302\n",
      "Data columns (total 14 columns):\n",
      " #   Column    Non-Null Count  Dtype  \n",
      "---  ------    --------------  -----  \n",
      " 0   age       303 non-null    int64  \n",
      " 1   sex       303 non-null    int64  \n",
      " 2   cp        303 non-null    int64  \n",
      " 3   trestbps  303 non-null    int64  \n",
      " 4   chol      303 non-null    int64  \n",
      " 5   fbs       303 non-null    int64  \n",
      " 6   restecg   303 non-null    int64  \n",
      " 7   thalach   303 non-null    int64  \n",
      " 8   exang     303 non-null    int64  \n",
      " 9   oldpeak   303 non-null    float64\n",
      " 10  slope     303 non-null    int64  \n",
      " 11  ca        303 non-null    int64  \n",
      " 12  thal      303 non-null    int64  \n",
      " 13  target    303 non-null    int64  \n",
      "dtypes: float64(1), int64(13)\n",
      "memory usage: 33.3 KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 284
    },
    "colab_type": "code",
    "id": "xMv_Nroqs9Uw",
    "outputId": "a4021027-f0d0-4a1f-8e82-6d3c381bb63a"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>sex</th>\n",
       "      <th>cp</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>fbs</th>\n",
       "      <th>restecg</th>\n",
       "      <th>thalach</th>\n",
       "      <th>exang</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>slope</th>\n",
       "      <th>ca</th>\n",
       "      <th>thal</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "      <td>303.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>54.366337</td>\n",
       "      <td>0.683168</td>\n",
       "      <td>0.966997</td>\n",
       "      <td>131.623762</td>\n",
       "      <td>246.264026</td>\n",
       "      <td>0.148515</td>\n",
       "      <td>0.528053</td>\n",
       "      <td>149.646865</td>\n",
       "      <td>0.326733</td>\n",
       "      <td>1.039604</td>\n",
       "      <td>1.399340</td>\n",
       "      <td>0.729373</td>\n",
       "      <td>2.313531</td>\n",
       "      <td>0.544554</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>9.082101</td>\n",
       "      <td>0.466011</td>\n",
       "      <td>1.032052</td>\n",
       "      <td>17.538143</td>\n",
       "      <td>51.830751</td>\n",
       "      <td>0.356198</td>\n",
       "      <td>0.525860</td>\n",
       "      <td>22.905161</td>\n",
       "      <td>0.469794</td>\n",
       "      <td>1.161075</td>\n",
       "      <td>0.616226</td>\n",
       "      <td>1.022606</td>\n",
       "      <td>0.612277</td>\n",
       "      <td>0.498835</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>29.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>94.000000</td>\n",
       "      <td>126.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>71.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>47.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>120.000000</td>\n",
       "      <td>211.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>133.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>55.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>130.000000</td>\n",
       "      <td>240.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>153.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.800000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>61.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>140.000000</td>\n",
       "      <td>274.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>166.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.600000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>77.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>200.000000</td>\n",
       "      <td>564.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>202.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>6.200000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>4.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              age         sex          cp    trestbps        chol         fbs  \\\n",
       "count  303.000000  303.000000  303.000000  303.000000  303.000000  303.000000   \n",
       "mean    54.366337    0.683168    0.966997  131.623762  246.264026    0.148515   \n",
       "std      9.082101    0.466011    1.032052   17.538143   51.830751    0.356198   \n",
       "min     29.000000    0.000000    0.000000   94.000000  126.000000    0.000000   \n",
       "25%     47.500000    0.000000    0.000000  120.000000  211.000000    0.000000   \n",
       "50%     55.000000    1.000000    1.000000  130.000000  240.000000    0.000000   \n",
       "75%     61.000000    1.000000    2.000000  140.000000  274.500000    0.000000   \n",
       "max     77.000000    1.000000    3.000000  200.000000  564.000000    1.000000   \n",
       "\n",
       "          restecg     thalach       exang     oldpeak       slope          ca  \\\n",
       "count  303.000000  303.000000  303.000000  303.000000  303.000000  303.000000   \n",
       "mean     0.528053  149.646865    0.326733    1.039604    1.399340    0.729373   \n",
       "std      0.525860   22.905161    0.469794    1.161075    0.616226    1.022606   \n",
       "min      0.000000   71.000000    0.000000    0.000000    0.000000    0.000000   \n",
       "25%      0.000000  133.500000    0.000000    0.000000    1.000000    0.000000   \n",
       "50%      1.000000  153.000000    0.000000    0.800000    1.000000    0.000000   \n",
       "75%      1.000000  166.000000    1.000000    1.600000    2.000000    1.000000   \n",
       "max      2.000000  202.000000    1.000000    6.200000    2.000000    4.000000   \n",
       "\n",
       "             thal      target  \n",
       "count  303.000000  303.000000  \n",
       "mean     2.313531    0.544554  \n",
       "std      0.612277    0.498835  \n",
       "min      0.000000    0.000000  \n",
       "25%      2.000000    0.000000  \n",
       "50%      2.000000    1.000000  \n",
       "75%      3.000000    1.000000  \n",
       "max      3.000000    1.000000  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "HvogvrIktNEf"
   },
   "source": [
    "#feature selection"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 1000
    },
    "colab_type": "code",
    "id": "LxvAZ0OVtQuW",
    "outputId": "8efc29dd-956c-40ca-bd5f-840b6a9f07c5"
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABCkAAARiCAYAAACJaa3IAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi41LCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvSM8oowAAIABJREFUeJzs3Xd8FNX6x/HP2d0USCekQCginYAgiAWUrqioWMCG4lUUy1W8yL12VBBQvOJV77WB7WcDpVdBRDAoUpUiHYFAgDRIgfTszu+PjQkhiQYh2Q1836+XL8nOmexzsjNnZp4586yxLAsREREREREREU+zeToAERERERERERFQkkJEREREREREvISSFCIiIiIiIiLiFZSkEBERERERERGvoCSFiIiIiIiIiHgFJSlERERERERExCsoSSEiIiIiIiIipRhjPjTGJBtjfq1guTHGvGmM2WWM2WiM6Xg63ldJChERERERERE50cfAlX+w/CqgedF/Q4F3TsebKkkhIiIiIiIiIqVYlhUHHPmDJv2BTyy3lUCoMabeqb6vkhQiIiIiIiIicrJigP3H/ZxQ9NopcZzqL/gz5sGLrap+D2/l7Bvr6RA8q7a/pyPwqIzu3TwdgkeF+kV6OgTPOrL/z9ucoVyLl3g6BI/a8/o6T4fgUU1nD/N0CB5lIpp7OgSPyX/rtMzyrbGy7xvo6RA8KiT5gKdD8KjciYs9HYJH1XpxvvF0DFXpjL2mfXfV/bgf0/jdRMuyJp7Ebyjvcz/lv1WVJylERERERERExLsUJSROJilxogSg4XE/NwAOnlJQ6HEPERERERERETl5c4DBRd/ycTGQYVnWoVP9pZpJISIiIiIiIiKlGGMmAz2AusaYBOB5wAfAsqx3gQXA1cAuIBu4+3S8r5IUIiIiIiIiIhUwtjO65EaFLMu67U+WW8DfT/f76nEPEREREREREfEKSlKIiIiIiIiIiFdQkkJEREREREREvIKSFCIiIiIiIiLiFVQ4U0RERERERKQCZ2vhTE/RTAoRERERERER8QpKUoiIiIiIiIiIV1CSQkRERERERES8gmpSiIiIiIiIiFRANSmql2ZSiIiIiIiIiIhXUJJCRERERERERLyCkhQiIiIiIiIi4hVUk0JERERERESkAqpJUb00k0JEREREREREvIKSFCIi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eFD07/qWZTmNMXlVEpmIiIiIiIiIh6kmRfWqbJLiFWC9MWYZ7m/36AaMM8YEAN9WUWwiIiIiIiIichap7Ld7fGCM+Rq4E9iG+1GPBMuysoB/VWF8IiIiIiIiInKWqOy3e9wLPAo0ANYDFwM/Ab2qLjQREREREREROZtU9nGPR4HOwErLsnoaY1oBo6ouLBERERERERHPU02K6lXZryDNtSwrF8AY42dZ1jagZdWFJSIiIiIiIiJnm8rOpEgwxoQCs4DFxpg04GDVhSUiIiIiIiIiZ5vKFs68oeifLxhjlgIhwMIqi0pEREREREREzjqVnUlRzLKs76siEBERERERERFvY4xqUlSnytakEBERERERERGpUkpSiIiIiIiIiIhXUJJCRERERERERLyCkhQiIiIiIiIi4hVOunCmiIiIiIiIyNnC2FQ4szppJoWIiIiIiIiIeAUlKURERERERETEKyhJISIiIiIiIiJeQTUpRERERERERCqgmhTVSzMpRERERERERMQrKEkhIiIiIiIiIl5BSQoRERERERER8QqqSSEiIiIiIiJSAdWkqF6aSSEiIiIiIiIiXkFJChERERERERHxCkpSiIiIiIiIiIhXUE0KERERERERkQqoJkX10kwKEREREREREfEKSlKIiIiIiIiIiFdQkkJEREREREREvIKSFCIiIiIiIiLiFaq8cKazb2xVv4XXsi/a7OkQPMp5y8WeDsGjQn780dMheFT+qn2eDsGjfG/t4ukQPMZEhHg6BI86d/wVng7Bo4z/2f35Wyk7PR2Cxzg6NvZ0CB4VcizL0yF4lGnQ3tMheFRB8mxPh+BRtTwdQBVT4czqpZkUIiIiIiIiIuIVlKQQEREREREREa+gJIWIiIiIiIiIeIUqr0khIiIiIiIiUlOpJkX10kwKEREREREREfEKSlKIiIiIiIiIiFdQkkJEREREREREvIJqUoiIiIiIiIhUQDUpqpdmUoiIiIiIiIiIV1CSQkRERERERES8gpIUIiIiIiIiIuIVVJNCREREREREpAKqSVG9NJNCRERERERERLyCkhQiIiIiIiIi4hWUpBARERERERERr6AkhYiIiIiIiIh4BRXOFBEREREREamAMSqcWZ00k0JEREREREREvIKSFCIiIiIiIiLiFZSkEBERERERERGvoJoUIiIiIiIiIhUwNtWkqE6a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iIiIiHgFJSlERERERERExCuocKaIiIiIiIhIBew2Vc6sTppJISIiIiIiIiJeQUkKEREREREREfEKSlKIiIiIiIiIiFdQTQoRERERERGRCtiNalJUp6pPUtT2r/K38FbOGzqBr4+nw/AY+5crPR2CR7kev9HTIXiUT36Bp0PwrIBQT0fgMaax5ekQPOpwTIynQ/Co9J4vejoEj2o8pJ2nQ/AYx41XeToEjyr4v9meDsGjjP/Zfe9z4wuXeToEj7rU0wHIGUWPe1SlszhBISIiIiIiInKylKQQEREREREREa9wds/LEhEREREREfkDdptqUlQnzaQQEREREREREa+gJIWIiIiIiIiIeAUlKURERERERETEK6gmhYiIiIiIiEgF7CpJUa00k0JEREREREREvIKSFCIiIiIiIiLiFZSkEBERERERERGvoCSFiIiIiIiIiHgFFc4UERERERERqYDdpsqZ1UkzKURERERERETEKyhJISIiIiIiIiJeQUkKEREREREREfEKqkkhIiIiIiIiUgG7UU2K6qSZFCIiIiIiIiLiFZSkEBERERERERGvoCSFiIiIiIiIiHgF1aQQERERERERqYBqUlQvzaQQEREREREREa+gJIWIiIiIiIiIeAUlKURERERERETEK6gmhYiIiIiIiEgF7Lq1X6305xYRERERERERr6AkhYiIiIiIiIh4BSUpRERERERERMQrKEkhIiIiIiIiIl5BhTNFREREREREKmA3xtMhnFU0k0JEREREREREvIKSFCIiIiIiIiLiFZSkEBERERERERGvoJoUIiIiIiIiIhWw21STojppJoWIiIiIiIiIeIUaNZPCsizGTd9B3OZU/H3tjLujDbENg8u0S0jNYcTHm0jPLqBNg2DGD47F1+HOx6zeeYSXpu+gwGkRFujDp49eAMAny/YxdcUBLAsGdonhrp6NqrVvlWFZFuO+3Ercrynu/v+tHbGNQsq0S0jNZsSk9e7+Nwxm/D3ti/sPsGlvOre+/BOv3deBvp3qAfDJkr1M/WG/u/+XNuCuPk2qrV9V4YM7n+Gadl1JPppGuxcHeTqc02L52gTGvrMSl8vFgCtbMvSW9qWWW5bF2HdWErdmP/5+Dl4a0Y3Y5nWLlzudLgYMm01keADvjb4CgK2/HeaF//5IXr4Tu93G8w934byWEdXar8qwLItxM3YS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TNNMA67g24QiIiIiIiIicrmVeSaFYRjBhckJDMOodhH7ioiIiIiIiFRK1kqwfs9fycW8gnSVYRizcKxFcRcQXW5RiYiIiIiIiMgVp0xJCtM0PzcMYwPQHcdimXeYprmzXCMTERERERERkStKmR/ZKExKKDEhIiIiIiIiIuVC60qIiIiIiIiInIPVojUpKlJZ3+4hIiIiIiIiIlKulKQQEREREREREbegJIWIiIiIiIiIuAWtSSEiIiIiIiJyDlZDa1JUJM2kEBERERERERG3oCSFiIiIiIiIiLgFJSlERERERERExC1oTQoRERERERGRc7Dq1n6F0tctIiIiIiIiIm5BSQoRERERERERcQtKUoiIiIiIiIiIW1CSQkRERERERETcghbOFBERERERETkHq2G4OoQrimZSiIiIiIiIiIhbUJJCRERERERERNyCkhQiIiIiIiIi4ha0JoWIiIiIiIjIOVi1JEWF0kwKEREREREREXELSlKIiIiIiIiIiFtQkkJERERERERE3ILWpBARERERERE5B4uhRSkqkmZSiIiIiIiIiIhbUJJCRERERERERNyCkhQiIiIiIiIi4ha0JoWIiIiIiIjIOVi1JEWF0kwKEREREREREXELSlKIiIiIiIiIiFso98c9TnbrWt6/wm0F/vqrq0NwKfuzd7g6BJey/Oc7V4fgUvboR10dgksZviGuDsFlbL+scnUILlXt9yRXh+BSId8+6OoQXMqo2dLVIbiMbdaXrg7BpTz/3sfVIbjWqQxXR+BS1/6wwdUhuNYwVwcgfyWaSSEiIiIiIiIibkELZ4qIiIiIiIicg0ULZ1YozaQQEREREREREbegJIWIiIiIiIiIuAUlKURERERERETELWhNChEREREREZFzsGpNigqlmRQiIiIiIiIi4haUpBARERERERERt6AkhYiIiIiIiIi4Ba1JISIiIiIiInIOFosWpahImkkhIiIiIiIiIm5BSQoRERERERERcQtKUoiIiIiIiIiIW9CaFCIiIiIiIiLnYNWSFBVKMylERERERERExC0oSSEiIiIiIiIibkFJChERERERERFxC0pSiIiIiIiIiIhb0MKZIiIiIiIiIudg0cKZFUozKURERERERETELShJISIiIiIiIiJuQUkKEREREREREXELWpNCRERERERE5BysWpOiQmkmhYiIiIiI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hCMiIiIiIiLiPqyGFqWoSGVJUniYpvnLmR8YhlG1nOIRERERERERkSvUOZMUhmE8AjwKNDQMY+sZm/yBX8s7MBERERERERG5spxvJsXXwA/AROBfZ3yeaZrmiXKNSkRERERERESuOOdMUpimeRI4CQy5lF8QVCXsUnav1PLWHnF1CC7lmZfv6hBcyh79qKtDcCnLC++7OgSXsj/a29UhuIz9ZK6rQ3Apj4Y+rg5BXMiM3+LqEFzG0vQKfxlcXo6rI3Atbz9XR+BSlrAru/5/dRYtSVGhzvsKUhERERERERGRiqIkhYiIiIiIiIi4BSUpRERERERERMQtlOUVpCIiIiIiIiJXJKvWpKhQmkkhIiIiIiIiIm5BSQoRERERERERcQtKUoiIiIiIiIiIW1CSQkRERERERETcghbOFBERERERETkHi6GVMyuSZlKIiIiIiIiIiFtQkkJERERERERE3IKSFCIiIiIiIiLiFrQmhYiIiIiIiMg5WLUkRYXSTAoRERERERERcQtKUoiIiIiIiIiIW1CSQkRERERERETcgtakEBERERERETkHi6FFKSqSZlKIiIiIiIiIiFtQkkJERERERERE3IKSFCIiIiIiIiLiFrQmhYiIiIiIiMg5aE2KiqWZFCIiIiIiIiLiFpSkEBERERERERG3oCSFiIiIiIiIiLgFJSlERERERERExC1o4UwRERERERGRc9DCmRVLMylERERERERExC0oSSEiIiIiIiIibkFJChERERERERFxC1qTQkREREREROQcLIbu7VckfdsiIiIiIiIi4haUpBARERERERERt6AkhYiIiIiIiIi4Ba1JISIiIiIiInIOFsNwdQhXlEqdpIiN2UF09LfY7XYGDrqeESN6O21fMH8tkyf/CICPbxVeeeVumjevDcDzoz9nxYpthIT4s2DhSxUee3mw3vwg1kYdMPNzKVj4X8ykgyXKeNw2EiOiMdgLMI/to2DxB2C3uSDai2eaJhO+20fMzuN4e1qYMDSSlnX8S5SLP36KUZ/tID07n8g6/rz290i8PCxMXRbHwo1JABTYTA4mZfNrdBfSsvJ46rMdRfsfTT3FP/o25L6oOhVWt4sVu/oA0W//iN1mMvC2qxhxb2en7QcPpzI6eiE79yTyz4eiGD70WgASkjJ4btx8Uo9nYbEY3NW/Hff+7RpXVKHcTL3n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uoO1qkghCMJEAMsAuEAaSSEAEEVRlO908ERERERERER/k6J1X4pJbaS1IyneBTBBFMXz7dkZIiIiIiIiIrpztbYkpGaBgoiIiIiIiIja001HUtQ95gEAxwRB+BFANICqP14XRfHnduwbEREREREREd1BbvW4x4S6/4oAKgCEN3pNBMAiBREREREREf1jKQTB0F24o9y0SCGK4uMAIAjCtwBeEEWxqO5newDL2797RERERERERHSnaO2cFL3/KFAAgCiKhQD6tk+XiIiIiIiIiOhO1NoihaJu9AQAQBAEB7T+m0GIiIiIiIiIiG6ptYWG5QAOCYKwCdJcFFMBLG63XhERERERERHJAOek6FitKlKIorhWEIRjAMIACAAmiqJ4rl17RkRERERERER3lFY/slFXlGBhgoiIiIiIiIjaRWvnpCAiIiIiIiIialcsUhARERERERGRLPAbOoiIiIiIiIhaoBB4b78j8dMmIiIiIiIiIllgkYKIiIiIiIiIZIFFCiIiIiIiIiKSBc5JQURERERERNQChSAYugt3FI6kICIiIiIiIiJZYJGCiIiIiIiIiGSBRQoiIiIiIiIikgXOSUFERERERETUAs5J0bE4koKIiIiIiIiIZIFFCiIiIiIiIiKSBRYpiIiIiIiIiEgWOCcFERERERERUQs4J0XH4kgKIiIiIiIiIpIFFimIiIiIiIiISBZYpCAiIiIiIiIiWWCRgoiIiIiIiIhkof0nziy42u5/Qq5MnpoIFGcbuhuGY2ln6B4YlGDpaOguGJT22bGG7oJBKT7faeguGIxmQh9Dd4EMqazM0D0wrOoaQ/fAYMTCYkN3waAUfr0M3QWD0p45ZuguGJaTvaF7QO1IIfDefkfip92e7uQCBREREREREdGfxCIFEREREREREckCixREREREREREJAvtPycFERERERER0W1KAcHQXbijcCQFEREREREREckCixREREREREREJAssUhARERERERGRLHBOCiIiIiIiIqIWKATOSdGROJKCiIiIiIiIiGSBRQoiIiIiIiIikgUWKYiIiIiIiIhIFjgnBREREREREVELFALv7XckftpEREREREREJAssUhARERERERGRLLBIQURERERERESywCIFEREREREREckCJ84kIiIiIiIiaoFCEAzdhTsKR1IQERERERERkSywSEFERERE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Ow3dBYPRTNCfnOxOIrjqz/9zR7G4fS+K2kR1jaF7YDBifuGtg/7BFAPkeaOno2jPyHyOq/amuLMvYhWDlxm+0tuOfi987x95TdvD/mVZLrfWfLvHEwCOApgIYDKAREEQZrV3x4iIiIiIiIjoztKaxz1eBtBXFMV8ABAEwRHAIQBftWfHiIiIiIiIiOjO0poiRSaA0kY/lwLgd9UQERERERHRP55CBvMW3UlaLFIIgjC/7n+vATgiCMJWSHNSREB6/IOIiIiIiIiIqM3cbE6KvgCsAUwAEA2pQAEAWwFktXO/iIiIiIiIiOgOc7PHPfoDeAPAJACfdEx3iIiIiIiIiOhOdbMixUoAOwH4AGj8nUICpFEVvu3YLyIiIiIiIiK6w7RYpBBF8WMAHwuC8D9RFJ/pwD4RERERERERyYIg3GyWBGprt/y0WaAgIiIiIiIioo7AkhARERERERERyQKLFEREREREREQkCzebOJOIiIiIiIjojqbgvf0OxU+biIiIiIiIiGSBRQoiIiIiIiIikgUWKYiIiIiIiIhIFjgnBREREREREVELBIH39jsSP20iIiIiIiIikgUWKYiIiIiIiIhIFlikICIiIiIiIiJZ4JwURERERERERC1QcE6KDsVPm4iIiIiIiIhkgUUKIiIiIiIiIpIFFimIiIiIiIiISBZYpCAiIiIiIiIiWeDEmUREREREREQtEHhvv0Px0yYiIiIiIiIiWWCRgoiIiIiIiIhkgUUKIiIiIiIiIpIFzklBRERERERE1AKFwHv7HYmfNhERERERERHJAosURERERERERCQLLFIQERERERERkSxwTgoiIiIiIiKiFgi8t9+h+GkTERERERERkSy0+0gK7a749v4TsiU42xq6CwYldBYN3QWD0uw7ZOguGJS2uMrQXTAozYQ+hu6CwRhtP2noLhiU9qUHDd0FgxIvXTF0FwxLaWToHhiMEBho6C4YlDZRZeguGJRgf4ef93YLMnQXiP4xOJKCiIiIiIiIiGSBc1IQERERERERtUAh8N5+R+KnTURERERERESywCIFEREREREREckCixREREREREREJAssUhARERERERGRLHDiTCIiIiIiIqIWCJw4s0Px0yYiIiIiIiIiWWCRgoiIiIiIiIhkgUUKIiIiIiIiIpIFzklBRERERERE1AIF7+13KH7aRERERERERCQLLFIQERERERERkSywSEFEREREREREssA5KYiIiIiIiIhaIAi8t9+R+GkTERERERERkSywSEFEREREREREssAiBRERERERERHJAuekICIiIiIiImqBgnNSdCh+2kREREREREQkCyxSEBEREREREZEssEhBRERERERERLLAOSmIiIiIiIiIWiDAyNBduKNwJAURERERERERyQKLFEREREREREQkCyxSEBEREREREZEssEhBRERERERERLLAiTOJiIiIiIiIWqAQeG+/I/HTJiIiIiIiIiJZYJGCiIiIiIiIiGSBRQoiIiIiIiIikgXOSUFERERERETUAoH39jsUP20iIiIiIiIikgUWKYiIiIiIiIhIFm7rxz1EUcSSHRlQpRTC3NgISx70Q6CHpV7cuiPZWHs4C1cLqnDw1f6wtzQGAGw/lYc1B64DACxMFHhrgg96uOm3lwtRFLHkp9+hOpsLMxMjLJkZhJ7eNnpxmXkVWLAmGUXlNQj0tsGyx4JgolTg6MUC/Pt/v6GTkzkAYHQfF/x7XFcAwDfx6dh0MBMCBHTztMKSmb1gamzUofndyv5jmVi8KhFarYjJY7phztRgnddFUcTiVUegSroKM1Mlls4fjp5dnepf12i0mPzCNrg4WmJV5D06bddsPo331iTh8PqHYW9r1iH5/B2iKGJJbDpUF+vW/Ul+CPSw0otbl5iFtYfq1v2FA+rX/bTcSiz6ORXnrpfjhXu8MWuYR0en8LeIooilCZnYn1YCM6WAxfd1QaCrhV7cDydy8N3xXFwtqsL+f/eGvYW0yzt6pRRzt1yCp60pAGB0Nzs8M8S9Q3P4M0RRxJItKVCdL4CZsQJLpgegp5e1XlxmfiUWrD2LoopaBHayxrIZATBRKlBaWYtXvj+HrKIbqNWImHW3NyYOashXoxUxZcUxuNiaYuWTvTsytTa35l+LMD5oKHJKCxH09gxDd6dN3On7PlEUsWTzRajO5knHvkcC0dOruWNfJRZ8cxpFFTUI7GSDZTN7wkSpwJrd6Yg5lg0AqNWKSMsux8GlobCzNMaidWex90weHKxNsP31uzo6tVtqz+P+2oQMbDyQCREipgzthEdHdenI1Fpl/5F0LP5kL7RaLSaP64U5MwbqvC6KIhZ/vBeqI5dhZmqMpQvD0bObKwBg7aYT2BhzBqIoYsr4IDw6pR8A4KM1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YwuAaZqphmF4VGBcIiIiIiIiIs5nVynGylTemRQFhmFYABPAMIwQQNVDRERERERERP6mDMO4wTCMvYZhHDAMo1TJB8Ph3dPbtxuG0fFif2Z5kxTvAvOAUMMwJuN4DOmUi/3hIiIiIiIiIuJ6Tk9UeB+4EWgFDDMMo9VZzW4EIk//Nwb48GJ/brmWe5imOcMwjE1AL8AABpimuftif7iIiIiIiIiIuKQrgQOmaR4CMAzja6A/sOuMNv2BL0zTNIF1hmEEGoZR2zTNuL/6Q8tbkwLTNPcAe/7qDxIRERERERGpcuzVttJBXeDYGV/HAp3L0aYu8JeTFOVd7iEiIiIiIiIifxOGYYwxDOO3M/4bc3aTMnY7u4poedpckHLPpBARERERERGRvwfTNKcCU8/TJBaof8bX9YATf6HNBdFMChERERERERE520Yg0jCMRoZheABDgQVntVkAjDj9lI8uQPrF1KMAzaQQEREREREROTf7Ra1eqLJM0yw0DONhYAlgAT4xTXOnYRj3n97+EbAI6AscAHKAey/25ypJISIiIiIiIiKlmKa5CEci4szXPjrj3ybw0KX8mVruISIiIiIiIiIuQUkKEREREREREXEJSlKIiIiIiIiIiEtQTQoRERERERGRc7HbnR1BtVLhSQojJLKif4TLMpP2OzsEp2owsq2zQ3Aqo05rZ4fgVGbsNmeH4Fz5Bc6OwHmsFmdH4FQevh7ODsGpkt961NkhOFWtce84OwSnmfFRqrNDcKqh79R3dghOZTg7ACdz0/x0kUtGbycRERERERERcQlKUoiIiIiIiIiIS1BNChEREREREZFzUU2KSqWZFCIiIiIiIiLiEpSkEBERERERERGXoCSFiIiIiIiIiLgE1aQQEREREREROQfTNJ0dQoVw1UcHayaFiIiIiIiIiLgEJSlERERERERExCUoSSEiIiIiIiIiLkE1KURERERERETOxW53dgTVimZSiIiIiIiIiIhLUJJCRERERERERFyCkhQiIiIiIiIi4hKUpBARERERERERl6DCmSIiIiIiIiLnosKZlUozKURERERERETEJShJISIiIiIiIiIuQUkKEREREREREXEJqkkhIiIiIiIici5209kRVCuaSSEiIiIiIiIiLkFJChERERERERFxCUpSiIiIiIiIiIhLUE0KERERERERkXOx250dQbWi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WTXjo1Rnh+A0w+8PcnYITjX0p9nODsGp7HmnnB2CU/mtfdjZIYj8bVR4kkJERERERESkqjLtqklRmbTcQ0RERERERERcgpIUIiIiIiIiIuISlKQQEREREREREZegJIWIiIiIiIiIuAQVzhQRERERERE5F5sKZ1YmzaQQEREREREREZegJIWIiIiIiIiIuAQlKURERERERETEJagmhYiIiIiIiMi52OzOjqBa0UwKEREREREREXEJSlKIiIiIiIiIiEtQkkJEREREREREXIJqUoiIiIiIiIicg2k3nR1CtaKZFCIiIiIiIiLiEpSkEBERERERERGXoCSFiIiIiIiIiLgE1aQQERERERERORebalJUJs2kEBERERERERGXoCSFiIiIiIiIiLgEJSlERERERERExCUoSSEiIiIiIiIiLkGFM0VERERERETOxa7CmZVJMylERERERERExCUoSSEiIiIiIiIiLkFJChERERERERFxCapJISIiIiIiInIOpk01KSqTZlKIiIiIiIiIiEtQkkJEREREREREXIKSFCIiIiIiIiLiElSTQkRERERERORc7HZnR1CtaCaFiIiIiIiIiLgEJSlERERERERExCUoSSEiIiIiIiIiLqFK16QwTZNXok+wKiYDT6sbL/WuT6tQ71LtZm5L5qttSRxLzyd6VGuCvBzd/nRzIj/uTQXAZodDqXlEj2pNgKdr/lpM02TKzN+J3p6Ap4eFKSM70LphYKl2sUnZPP7RJtKyCmjVIIBXx3TEw+rGss1xvDtvD26GgcViMH5YGy5vFgzAs9O3sGJbAjX9a7DwpWsru2sXbNXag0x++3/YbSaDb7mMMSO6lth+6Egy4yf/wK698fzzH9cwcngXAOISMnhq0gKSU7JwczO4rX8HRtx+pTO6cFFM02TKohii96fi5W5hyq1NaFXHp1S7Gevj+WJtHMdOnuLXpy4nyMcdgIXbkpm++gQA3h5uPN+vES3CS+/vKkzTZPLnW4jeEo9nDQsvP3AlrRsFlWoXm5jFY++sIz07n1YNg3j14SvxsFr+dH+b3c7gZ5YSGuTFx091r8yuXTDTNJny7R6idyY5zgMj2tI6wr9Uu9jkHB6fvp207AJaRfjz6j1t8bA68tIb9p3k5dl7KLDZCfL14MvHXPs9sGr9ESb/ZwV2u53BN7VhzPCS8ZqmyeR3VxC9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4KgcNklhjOl5yFM7S/9taoxp+uftRERERERERESOS06XpyNoUI40kuLZSp47uGrIwBqMRUREREREREQasMMmKay1AwCMMaOAL6y1OcaYB4GewKQ6iE9EREREREREGojqFs6cUJqgOBN3UYy3gP/VWlQiIiIiIiIi0uBUN0nhLP33AuAVa+0swL92QhIRERERERGRhqhad/cAdhljXgXOBp40xgRQ/QSHiIiIiIiISL1kXfbIjaTGVDfRMAr4EjjPWpsFRAF311pUIiIiIiIiItLgVGskhbU2H/jkoMfJQHJtBSUiIiIiIiIiDY+mbIiIiIiIiIiIV6huTQoRERERERGRhsepmhR1SSMpRERERERERMQrKEkhIiIiIiIiIl5BSQoRERERERER8QqqSSEiIiIiIiJSFZdqUtQljaQQEREREREREa+gJIWIiIiIiIiIeAUlKURERERERETEK6gmhYiIiIiIiEgVrFM1KeqSRlKIiIiIiIiIiFdQkkJEREREREREvIKSFCIiIiIiIiLiFYy1tTu/JqvwowY7gSc8N8/TIXhU8TuzPR2CR/ldfr6nQ/CsonxPR+BRJjTO0yF4jGtZkqdD8KiiZds9HYJHBdwy3NMheJSNauHpEDzn8488HYFH+cxe5ekQPMo+fbenQ/Cozec94ukQPCphyRrj6RhqU/695x+X17RBT3zule+bCmeKiIiIiIiIVMV1XOYovJame4iIiIiIiIiIV1CSQkRERERERES8gpIUIiIiIiIiIuIVVJNCREREREREpCpOl6cjaFA0kkJEREREREREvIKSFCIiIiIiIiLiFZSkEBERERERERGvoJoUIiIiIiIiIlWwLuvpEBoUjaQQEREREREREa+gJIWIiIiIiIiIeAUlKURERERERETEK6gmhYiIiIiIiEhVnKpJUZc0kkJEREREREREvIKSFCIiIiIiIiLiFZSkEBERERERERGvoCSFiIiIiIiIiHgFFc4UERERERERqYJ1qXBmXdJIChERERERERHxCkpSiIiIiIiIiIhXUJJCRERERERERLyCalKIiIiIiIiIVME6VZOiLmkkhYiIiIiIiIh4BSUpRERERERERMQrKEkhIiIiIiIiIl5BNSlEREREREREqmBdqklRlzSSQkRERERERES8gpIUIiIiIiIiIuIVlKQQEREREREREa+gmhQi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AouBFsCLQBjwiLX2iHu/2kxSeLvaTFLUB7WdpPB2dZWk8Fo1mKSoj2orSVEf1HSSor6p7SSFt6vNJEV9UFtJinqhFpMU9YEnkxTeoDaTFPVBbScpvN3xnqTYc8WZx+U1bfS7S7zyfavudI+RuBMav1prBwDnAJfUXlgiIiIiIiIi0tBUN0lxgrW2bMK/tXYvUH8qL4mIiIiIiIiI16tuksJhjCm7QbcxJorq3xlEREREREREROSIqptoeBZYaoyZgbtM+Chgcq1FJSIiIiIiIuIFrOu4LEnhtaqVpLDWvmOM+REYCBjgUmvt2lqNTEREREREREQalGpP2ShNSigxISIiIiIiIiK1oro1KUREREREREREapWSFCIiIiIiIiLiFXSHDhEREREREZEqWKcKZ9YljaQQEREREREREa+gJIWIiIiIiIiIeAUlKURERERERETEK6gmhYiIiIiIiEgVrEs1KeqSRlKIiIiIiIiIiFdQkkJEREREREREvCj1vIoAACAASURBVIKSFCIiIiIiIiLiFVSTQkRERERERKQKLtWkqFMaSSEiIiIiIiIiXkFJChERERERERHxCkpSiIiIiIiIiIhXUE0KERERERERkSpYp2pS1CWNpBARERERERERr6AkhYiIiIiIiIh4BSUpRERERERERMQrKEkhIiIiIiIiIl5BhTNFREREREREqmBdKpxZlzSSQkRERERERES8gpIUIiIiIiIiIuIVlKQQEREREREREa+gmhQiIiIiIiIiVVBNirqlkRQiIiIiIiIi4hWUpBARERERERERr6AkhYiIiIiIiIh4BdWkEBEREREREamCdaomRV2q9SRFeNqu2v4TXss07+HpEDzKBDbwHFhBjqcj8KzAEE9H4FGuX3/0dAgeYyLDPR2CR/mfGOfpEDzKRLXydAgeZTwdgAe59hd6OgSPsk/f7ekQPMrc/bSnQ/ConZGBng5B5Lih6R4iIiIiIiIi4hWUpBARERERERERr9DAx+OLiIiIiIiIVM26XJ4OoUHRSAoRERERERER8QpKUoiIiIiIiIiIV1CSQkRERERERES8gpIUIiIiIiIiIuIVVDhTREREREREpArWaT0dQoOikRQiIiIiIiIi4hWUpBARERERERERr6AkhYiIiIiIiIh4BdWkEBEREREREamCdakmRV3SSAoRERERERER8QpK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IiIiIiIixwOX67ic7eG1qjOSAmNMH6D1n+2NMVhrG/bEQxERERERERGpUUdMUhhj3gUSgFXAnxVhLKAkhYiIiIiIiIjUmOqMpDgZ6GKPdBsQEREREREREZFjUJ0kxa9APJBcy7GIiIiIiIiIeBWXy9MRNCxVJimMMbNxT+sIBdYaY1YAhX8ut9ZeVPvhiYiIiIiIiEhDcbiRFM8ABngSuPig5/98TkRERERERESkxlSZpLDWLgIwxvj9+fufjDGNajswEREREREREWlYDjfd4ybgZqCtMWb1QYtCge9qOzARERERERERT1NNirp1uOke7wOfA48D9x70/D5r7d5ajUpEREREREREGpzDTffIBrKBy+ouHBERERERERFpqByeDkBEREREREREBJSkEBEREREREREvcbiaFCIiIiIiIiINmgpn1i2NpBARERERERERr6AkhYiIiIiIiIh4BSUpRERERERERMQrqCaFiIiIiIiISBVc1tMRNCwaSSEiIiIiIiIiXkFJChERERERERHxCkpSiIiIiIiIiIhXUE0KERERERERkSq4XJ6OoGHRSAoRERERERER8QpKUoiIiIiIiIiIV1CSQkRERERERES8gmpSiIiIiIiIiFRBNSnqlkZSiIiIiIiIiIhXUJJCRERERERERLyCkhQiIiIiIiIi4hWUpBARERERERERr6DCmSIiIiIiIiJVUOHMuqWRFCIiIiIiIiLiFWp9JMX+KV/X9p/wWsVpszwdgketfqSvp0PwqNM+/9HTIXiUIzbE0yF4VuNIT0fgMaZDd0+H4FHLW2zwdAge1ajrTZ4OwaMcDfjrn9Dv/+HpEDzKed4jng7Bo3ZGBno6BI9q3tzH0yF4lPV0AHJcacCHUhERERERERHxJqpJISIiIiIiIlIF1aSoWxpJISIiIiIiIiJeQUkKEREREREREfEKSlKIiIiIiIiIiFdQTQoRERERERGRKqgmRd3SSAoRERERERER8QpKUoiIiIiIiIiIV1CSQkRERERERES8gmpSiIiIiIiIiFRBNSnqlkZSiIiIiIiIyP+3d99xUlV3H8c/v9nC9ga79K5IFQWsCAIaUezBxIIan2iMsUVF88QKIfZgTNQYS0ysqI9Yo4goCNgABaUoIIogICxLL9tnzvPHvdvYWVhgd2eW/b5fr33tzNwzs7/fvXfPPffcc8+IRAV1UoiIiIiIiIhIVFAnhYiIiIiIiIhEBXVSiIiIiIiIiEhU0MSZIiIiIiIiIjXQxJkNSyMpRERERERERCQqqJNCRERERERERKKCOilEREREREREJCpoTgoRERERERGRGmhOioalkRQiIiIiIiIiEhXUSSEiIiIiIiIiUUGdFCIiIiIiIiISFTQnhYiIiIiIiEgNnHORDqFJ0UgKEREREREREYkK6qQQERERERERkaigTgoRERERERERiQqak0JERERERESkBqFQpCNoWjSSQkRERERERESigjopRERERERERCQqqJNCRERERERERKKCOilEREREREREJCpo4kwRERERERGRGmjizIalkRQiIiIiIiIiEhXUSSEiIiIiIiIiUaHR3+4RN+K3BLoNgJIiil97ELf2++plzrmRQNuDIVhKaPW3lLz1CISCBLofTdwJF4JzEApSMukJQj9+E4Es9k2zc68hrvdRuOJCCp6+j9CqZdXKJP76VgIdu0EwSHDFEgqffwBCQUhIJvHSWwhktoSYGIrff5mSTydHIIt9s3D2Wl585Etc0DHo1C6MGNUjbLkflmzkriuncsUdxzBgSHtKioLc9/tplJQECQUd/Y9vz1n/07uBo99/zjnumbqKmd9vUoKzMgAAIABJREFUIzEuwF0jOtGzVVK1ci/MXc9zX6xn1ZYiPr6mL5lJVf/lF67dyQXPLWH8GV0Y3j2zocLfa8457n7je2Yu3khCfAx3n3cIvdqlViu3emMBo59fzJb8Unq2TeG+C7oTHxtge0Epf5iwmLWbiygNOX49pD0/P7IVACfcOYvkZrHEBCAmYEy8vn9Dp7dHzjnufmEhM+fnevn/ph+9OmVUK7c6byejH/2CLTuL6dkxg/t+25/42AD//XQV/3rHqx+SEmIZ86u+dO+QDsDTk79j4oyVmEG3dmncfVk/msXHNGh+e2PmZ8u464HJhEIhfnFmPy7/1aAqy79fkcct497k66Vruf53w7j0woFVlgeDIUb+6glaZqfy+IOjGjL0OrFg9k+88PA8QiHH8ad25bRRPcOWW754I+OufJ+rxhzLEUM6sHH9Tp64axZbNxViARh6+kGcdM4hDRx93Whzy62kDR5MqKCQVbfcTMHimo/bbW+9jcyzz2bRgKr/14m9e3Pwiy+zcvQNbJ3yXn2HXKda33wrqYMGEyos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\n",
      "text/plain": [
       "<Figure size 1440x1440 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "corrmat=df.corr()\n",
    "top_corr_features=corrmat.index\n",
    "plt.figure(figsize=(20,20))\n",
    "g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap='RdYlGn')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 566
    },
    "colab_type": "code",
    "id": "EKFflUZZLOdc",
    "outputId": "fe7bf2f2-7e68-41dc-fe87-6f4d8b7c339e"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[<matplotlib.axes._subplots.AxesSubplot object at 0x0000026A75B1C288>,\n",
       "        <matplotlib.axes._subplots.AxesSubplot object at 0x0000026A77C65888>,\n",
       "        <matplotlib.axes._subplots.AxesSubplot object at 0x0000026A77C965C8>,\n",
       "        <matplotlib.axes._subplots.AxesSubplot object at 0x0000026A77CC5308>],\n",
       "       [<matplotlib.axes._subplots.AxesSubplot object at 0x0000026A77CF7048>,\n",
       "        <matplotlib.axes._subplots.AxesSubplot object at 0x0000026A77CF7108>,\n",
       "        <matplotlib.axes._subplots.AxesSubplot object at 0x0000026A77D1ED88>,\n",
       "        <matplotlib.axes._subplots.AxesSubplot object at 0x0000026A77D7F8C8>],\n",
       "       [<matplotlib.axes._subplots.AxesSubplot object at 0x0000026A77DB1608>,\n",
       "        <matplotlib.axes._subplots.AxesSubplot object at 0x0000026A77DDF348>,\n",
       "        <matplotlib.axes._subplots.AxesSubplot object at 0x0000026A77E08F48>,\n",
       "        <matplotlib.axes._subplots.AxesSubplot object at 0x0000026A77E39C88>],\n",
       "       [<matplotlib.axes._subplots.AxesSubplot object at 0x0000026A77E6B9C8>,\n",
       "        <matplotlib.axes._subplots.AxesSubplot object at 0x0000026A77E9B708>,\n",
       "        <matplotlib.axes._subplots.AxesSubplot object at 0x0000026A77ECB448>,\n",
       "        <matplotlib.axes._subplots.AxesSubplot object at 0x0000026A77EFA248>]],\n",
       "      dtype=object)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 16 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.hist()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 296
    },
    "colab_type": "code",
    "id": "d4JWwJaULVWS",
    "outputId": "5a31411e-70c0-4096-a59f-2ee5775d472d"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x26a7803fec8>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set_style('whitegrid')\n",
    "sns.countplot(x='target',data=df,palette='RdBu_r')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "Y6hfjuWXQsra"
   },
   "source": [
    "# #it's a balances dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "JPQKOphhQePC"
   },
   "outputs": [],
   "source": [
    "dataset=pd.get_dummies(df,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "zMewvcq2kvJ-"
   },
   "source": [
    "this is to scale the features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "VDZ60n1njpGg"
   },
   "outputs": [],
   "source": [
    "standardScaler=StandardScaler()\n",
    "columns_to_scale=['age','trestbps','chol','thalach','oldpeak']\n",
    "dataset[columns_to_scale]=standardScaler.fit_transform(dataset[columns_to_scale])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 215
    },
    "colab_type": "code",
    "id": "Wi_JRNq-l8KC",
    "outputId": "fa57f463-b05e-4c04-8b7a-9b44ca768522"
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>trestbps</th>\n",
       "      <th>chol</th>\n",
       "      <th>thalach</th>\n",
       "      <th>oldpeak</th>\n",
       "      <th>target</th>\n",
       "      <th>sex_0</th>\n",
       "      <th>sex_1</th>\n",
       "      <th>cp_0</th>\n",
       "      <th>cp_1</th>\n",
       "      <th>...</th>\n",
       "      <th>slope_2</th>\n",
       "      <th>ca_0</th>\n",
       "      <th>ca_1</th>\n",
       "      <th>ca_2</th>\n",
       "      <th>ca_3</th>\n",
       "      <th>ca_4</th>\n",
       "      <th>thal_0</th>\n",
       "      <th>thal_1</th>\n",
       "      <th>thal_2</th>\n",
       "      <th>thal_3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0.952197</td>\n",
       "      <td>0.763956</td>\n",
       "      <td>-0.256334</td>\n",
       "      <td>0.015443</td>\n",
       "      <td>1.087338</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>-1.915313</td>\n",
       "      <td>-0.092738</td>\n",
       "      <td>0.072199</td>\n",
       "      <td>1.633471</td>\n",
       "      <td>2.122573</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>-1.474158</td>\n",
       "      <td>-0.092738</td>\n",
       "      <td>-0.816773</td>\n",
       "      <td>0.977514</td>\n",
       "      <td>0.310912</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0.180175</td>\n",
       "      <td>-0.663867</td>\n",
       "      <td>-0.198357</td>\n",
       "      <td>1.239897</td>\n",
       "      <td>-0.206705</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0.290464</td>\n",
       "      <td>-0.663867</td>\n",
       "      <td>2.082050</td>\n",
       "      <td>0.583939</td>\n",
       "      <td>-0.379244</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 31 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        age  trestbps      chol   thalach   oldpeak  target  sex_0  sex_1  \\\n",
       "0  0.952197  0.763956 -0.256334  0.015443  1.087338       1      0      1   \n",
       "1 -1.915313 -0.092738  0.072199  1.633471  2.122573       1      0      1   \n",
       "2 -1.474158 -0.092738 -0.816773  0.977514  0.310912       1      1      0   \n",
       "3  0.180175 -0.663867 -0.198357  1.239897 -0.206705       1      0      1   \n",
       "4  0.290464 -0.663867  2.082050  0.583939 -0.379244       1      1      0   \n",
       "\n",
       "   cp_0  cp_1  ...  slope_2  ca_0  ca_1  ca_2  ca_3  ca_4  thal_0  thal_1  \\\n",
       "0     0     0  ...        0     1     0     0     0     0       0       1   \n",
       "1     0     0  ...        0     1     0     0     0     0       0       0   \n",
       "2     0     1  ...        1     1     0     0     0     0       0       0   \n",
       "3     0     1  ...        1     1     0     0     0     0       0       0   \n",
       "4     1     0  ...        1     1     0     0     0     0       0       0   \n",
       "\n",
       "   thal_2  thal_3  \n",
       "0       0       0  \n",
       "1       1       0  \n",
       "2       1       0  \n",
       "3       1       0  \n",
       "4       1       0  \n",
       "\n",
       "[5 rows x 31 columns]"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "nJ-wzWBkmJzM"
   },
   "outputs": [],
   "source": [
    "y=dataset['target']\n",
    "X=dataset.drop(['target'],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "p6xH9VVImkM9"
   },
   "outputs": [],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "knn_scores=[]\n",
    "for k in range(1,21):\n",
    "  knn_classifier=KNeighborsClassifier(n_neighbors=k)\n",
    "  score=cross_val_score(knn_classifier,X,y,cv=10)\n",
    "  knn_scores.append(score.mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 295
    },
    "colab_type": "code",
    "id": "yu_YGQw_UBH2",
    "outputId": "5a0740f6-09af-4ede-9d94-a72d6a391b53"
   },
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot([k for k in range(1,21)],knn_scores,color='red')\n",
    "for i in range(1,21):\n",
    "  plt.text(i,knn_scores[i-1],(i,knn_scores[i-1]))\n",
    "  plt.xticks([i for i in range(i,21)])\n",
    "   \n",
    "  plt.xlabel('Number of Neighbors (k)')\n",
    "  plt.ylabel('Scores')\n",
    "  plt.title('K Neighbors Classifier scores for different K values')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 353
    },
    "colab_type": "code",
    "id": "rb5AS-w-azki",
    "outputId": "cecbf147-b86b-48f1-b404-83e5e5a9a0a8"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[0.7482684464219502,\n",
       " 0.7652651093807934,\n",
       " 0.8112940304041528,\n",
       " 0.8046273637374861,\n",
       " 0.8338820912124584,\n",
       " 0.8307563959955505,\n",
       " 0.8374378939562476,\n",
       " 0.8410938079347423,\n",
       " 0.8474378939562477,\n",
       " 0.8474378939562477,\n",
       " 0.8407638116425659,\n",
       " 0.8506637004078605,\n",
       " 0.8338820912124583,\n",
       " 0.8337671486837227,\n",
       " 0.8272080088987763,\n",
       " 0.8372154245457917,\n",
       " 0.8274304783092326,\n",
       " 0.8209788654060068,\n",
       " 0.8242046718576195,\n",
       " 0.8243196143863551]"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "knn_scores"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 353
    },
    "colab_type": "code",
    "id": "mgmXYeKzY96J",
    "outputId": "ff2f12fa-a04f-47a7-b09a-52ea676c9b29"
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "1 0.7482684464219502\n",
      "2 0.7652651093807934\n",
      "3 0.8112940304041528\n",
      "4 0.8046273637374861\n",
      "5 0.8338820912124584\n",
      "6 0.8307563959955505\n",
      "7 0.8374378939562476\n",
      "8 0.8410938079347423\n",
      "9 0.8474378939562477\n",
      "10 0.8474378939562477\n",
      "11 0.8407638116425659\n",
      "12 0.8506637004078605\n",
      "13 0.8338820912124583\n",
      "14 0.8337671486837227\n",
      "15 0.8272080088987763\n",
      "16 0.8372154245457917\n",
      "17 0.8274304783092326\n",
      "18 0.8209788654060068\n",
      "19 0.8242046718576195\n",
      "20 0.8243196143863551\n"
     ]
    }
   ],
   "source": [
    "for i in range(0,len(knn_scores)):\n",
    "  print (i+1,knn_scores[i])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "gnsAeVtQYYby"
   },
   "outputs": [],
   "source": [
    "knn_classifier=KNeighborsClassifier(n_neighbors=12)\n",
    "score=cross_val_score(knn_classifier,X,y,cv=10)\n",
    "  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 34
    },
    "colab_type": "code",
    "id": "4eLhj5bQaLO3",
    "outputId": "37261869-b23c-44b7-c89d-3efb83e9cb0c"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8506637004078605"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score.mean()#accuracy of k nearest neighbors"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pickle.dump(knn_classifier,open('model.pkl','wb'))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "model=pickle.load(open('model.pkl,'rb'))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "colab_type": "text",
    "id": "Dy3xVeR5cMn1"
   },
   "source": [
    "Random forest classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "sDaMgSVPcRjI"
   },
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "aPpbr3Mxco8F"
   },
   "outputs": [],
   "source": [
    "randomforest_classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=1)\n",
    "score=cross_val_score(randomforest_classifier,X,y,cv=10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 34
    },
    "colab_type": "code",
    "id": "Re4w2euwc-Cc",
    "outputId": "54e3d725-91f2-454f-8610-6b054df06ce4"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.823874675565443"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score.mean() #accuracy by random forest classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "colab": {},
    "colab_type": "code",
    "id": "bcQhwIlzdXxv"
   },
   "outputs": [],
   "source": [
    "decisiontreeclassifier=DecisionTreeClassifier(criterion='gini')\n",
    "score=cross_val_score(DecisionTreeClassifier(),X,y,cv=10)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/",
     "height": 34
    },
    "colab_type": "code",
    "id": "vPfDtRztj3KU",
    "outputId": "51761925-fb7e-445d-ece0-3918a294731a"
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.751924360400445"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "score.mean()#accuracy by decision tree classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "Untitled1.ipynb",
   "provenance": []
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
