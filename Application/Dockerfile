# pull python base image
FROM python:3.10

# specify working directory
WORKDIR /heart_model

ADD /heart_model/requirements.txt .
ADD /heart_model/*.pkl .

# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

#RUN rm *.whl

# copy application files
ADD /heart_model/* ./app/

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/app.py"]
