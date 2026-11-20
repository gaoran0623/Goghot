#include "gi_server.h"

#include "gi_node.h"

#include <godot_cpp/classes/engine.hpp>
#include <godot_cpp/classes/rendering_server.hpp>
#include <godot_cpp/classes/scene_tree.hpp>
#include <godot_cpp/classes/window.hpp>

using namespace godot;

namespace goghot {

void GoghotGIServer::tick() {
	SceneTree *tree = Object::cast_to<SceneTree>(Engine::get_singleton()->get_main_loop());
	if (tree == nullptr) {
		return;
	}
	Node *root = Engine::get_singleton()->is_editor_hint() ? tree->get_edited_scene_root() : tree->get_root();
	GoghotGI *node = node_id.is_valid() ? Object::cast_to<GoghotGI>(ObjectDB::get_instance(uint64_t(node_id))) : nullptr;
	if (root == nullptr) {
		return;
	}
	if (node != nullptr && root->get_instance_id() == root_id) {
		return;
	}
	if (node != nullptr) {
		node->queue_free(); // root changed (another edited scene): follow it
	}
	node = memnew(GoghotGI);
	node->set_name("GoghotGIEnvironment");
	node->set_environment_mode(true);
	root->call_deferred("add_child", node, false, Node::INTERNAL_MODE_BACK);
	node_id = ObjectID(node->get_instance_id());
	root_id = root->get_instance_id();
}

void GoghotGIServer::connect_late() {
	if (!Engine::get_singleton()->has_singleton("RenderingServer")) {
		callable_mp(this, &GoghotGIServer::connect_late).call_deferred();
		return;
	}
	RenderingServer::get_singleton()->connect("frame_pre_draw", callable_mp(this, &GoghotGIServer::tick));
}

void GoghotGIServer::shutdown() {
	node_id = ObjectID();
	root_id = 0;
}

} // namespace goghot
