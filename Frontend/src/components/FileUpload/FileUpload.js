import React from "react";
import { URLContext } from "../../contexts/URLcontext";

class FileUploader extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      videoName: "",
    };
  }
  handlevideoname = (e) => {
    this.setState({ videoName: e.target.value });
  };

  render() {
    return (
      <URLContext.Consumer>
        {(context) => {
          const { PythonURL, setVideoURL, setVideoName } = context;
          const handleUploadVideo = (ev) => {
            ev.preventDefault();

            const data = new FormData();
            data.append("file", this.uploadInput.files[0]);
            data.append("filename", this.fileName.value);

            fetch(`${PythonURL}/api/fileUpload`, {
              method: "POST",
              body: data,
            }).then((response) => {
              response.json().then(
                (body) => (
                  setVideoURL(
                    `https://t.zerxbot.workers.dev/0:/${body.filename}`,
                    () => {
                      window.open(`${PythonURL}/home`, "_self");
                    }
                  ),
                  setVideoName(body.filename)
                )
              );
            });
          };
          return (
            <form onSubmit={handleUploadVideo}>
              <div>
                <input
                  ref={(ref) => {
                    this.uploadInput = ref;
                  }}
                  type="file"
                />
              </div>
              <div>
                <input
                  ref={(ref) => {
                    this.fileName = ref;
                  }}
                  type="text"
                  placeholder="Enter the desired name of file"
                  onChange={this.handlevideoname}
                />
              </div>
              <br />
              <div>
                <button>Upload</button>
              </div>
            </form>
          );
        }}
      </URLContext.Consumer>
    );
  }
}

export default FileUploader;
