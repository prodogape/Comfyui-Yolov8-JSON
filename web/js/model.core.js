import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'
import { applyTextReplacements } from "../../../scripts/utils.js";

function chainCallback(object, property, callback) {
    if (object == undefined) {
        //This should not happen.
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r
        };
    } else {
        object[property] = callback;
    }
}

function injectHidden(widget) {
    widget.computeSize = (target_width) => {
        if (widget.hidden) {
            return [0, -4];
        }
        return [target_width, 20];
    };
    widget._type = widget.type
    Object.defineProperty(widget, "type", {
        set: function (value) {
            widget._type = value;
        },
        get: function () {
            if (widget.hidden) {
                return "hidden";
            }
            return widget._type;
        }
    });
}

async function uploadFile(file) {
    //TODO: Add uploaded file to cache with Cache.put()?
    try {
        // Wrap file in formdata so it includes filename
        const body = new FormData();
        const i = file.webkitRelativePath.lastIndexOf('/');
        const subfolder = file.webkitRelativePath.slice(0, i + 1)
        const new_file = new File([file], file.name, {
            type: file.type,
            lastModified: file.lastModified,
        });
        body.append("image", new_file);
        if (i > 0) {
            body.append("subfolder", subfolder);
        }
        const resp = await api.fetchApi("/upload/image", {
            method: "POST",
            body,
        });

        if (resp.status === 200) {
            return resp.status
        } else {
            alert(resp.status + " - " + resp.statusText);
        }
    } catch (error) {
        alert(error);
    }
}

function addUploadWidget(nodeType, nodeData, widgetName, type = "model") {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const pathWidget = this.widgets.find((w) => w.name === widgetName);
        const fileInput = document.createElement("input");
        chainCallback(this, "onRemoved", () => {
            fileInput?.remove();
        });
        if (type == "model") {
            Object.assign(fileInput, {
                type: "file",
                accept: ".pt",
                style: "display: none",
                onchange: async () => {
                    if (fileInput.files.length) {
                        if (await uploadFile(fileInput.files[0]) != 200) {
                            //upload failed and file can not be added to options
                            return;
                        }
                        const filename = fileInput.files[0].name;
                        pathWidget.options.values.push(filename);
                        pathWidget.value = filename;
                        if (pathWidget.callback) {
                            pathWidget.callback(filename)
                        }
                    }
                },
            });
        } else {
            throw "Unknown upload type"
        }
        document.body.append(fileInput);
        let uploadWidget = this.addWidget("button", "choose " + type + " to upload", "image", () => {
            //clear the active click event
            app.canvas.node_widget = null
            fileInput.click();
        });
        uploadWidget.options.serialize = false;
    });
}

function addCustomLabel(nodeType, nodeData, widgetName = "detect") {
    //Add a callback which sets up the actual logic once the node is created
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const node = this;
        const sizeOptionWidget = node.widgets.find((w) => w.name === widgetName);
        const LabelName = node.widgets.find((w) => w.name === "label_name");
        const LabelList = node.widgets.find((w) => w.name === "label_list");
        injectHidden(LabelName);
        injectHidden(LabelList);
        sizeOptionWidget._value = sizeOptionWidget.value;
        Object.defineProperty(sizeOptionWidget, "value", {
            set: function (value) {
                //TODO: Only modify hidden/reset size when a change occurs
                if (value === "choose") {
                    LabelName.hidden = true;
                    LabelList.hidden = false;
                } else if (value === "input") {
                    LabelName.hidden = false;
                    LabelList.hidden = true;
                } else {
                     LabelName.hidden = true;
                     LabelList.hidden = true;
                }
                node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]])
                this._value = value;
            },
            get: function () {
                return this._value;
            }
        });
        sizeOptionWidget.value = sizeOptionWidget._value;
    });
}
app.registerExtension({
    name: "Yolov8HelperSuite.Core",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData ?.name == "Apply Yolov8 Model") {
            addCustomLabel(nodeType, nodeData, "detect")
        }
        // else if (nodeData ?.name == "Load Yolov8 Model Upload") {
        //     addUploadWidget(nodeType, nodeData, "model");
        //     chainCallback(nodeType.prototype, "onNodeCreated", function () {
        //         const pathWidget = this.widgets.find((w) => w.name === "model_path");
        //         chainCallback(pathWidget, "callback", (value) => {
        //             if (!value) {
        //                 return;
        //             }
        //             let params = {
        //                 filename: value,
        //                 type: "input",
        //                 format: "folder"
        //             };
        //             this.updateParameters(params, true);
        //         });
        //     });
        // }
    }
});