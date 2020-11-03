import React from "react";
import { URLContext } from "../../contexts/URLcontext";

class FileUploader extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      imageURL: "",
    };
  }

  render() {
    return (
      <URLContext.Consumer>
        {(context) => {
          const { PythonURL } = context;
          const handleUploadVideo = (ev) => {
            ev.preventDefault();

            const data = new FormData();
            data.append("file", this.uploadInput.files[0]);
            data.append("filename", this.fileName.value);

            fetch(`${PythonURL}/api/fileUpload`, {
              method: "POST",
              body: data,
            }).then(() => {
              window.open(`${PythonURL}/home`, "_self");
            });
            // .then((response) => {
            //   response.json().then((body) => {
            //     this.setState(
            //       {
            //         imageURL: `${PythonURL}/api/video/${body.filename}`,
            //       },
            //       () => {
            //         console.log(`uplaod completed`);
            //       }
            //     );
            //   });
            // });
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
