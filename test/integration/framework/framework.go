/*
Copyright 2024 The Kubeflow Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package framework

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"path/filepath"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/webhook"

	kubeflowv2 "github.com/kubeflow/training-operator/pkg/apis/kubeflow.org/v2alpha1"
	controllerv2 "github.com/kubeflow/training-operator/pkg/controller.v2"
	webhookv2 "github.com/kubeflow/training-operator/pkg/webhook.v2"
)

type Framework struct {
	testEnv *envtest.Environment
	cancel  context.CancelFunc
}

func (f *Framework) Init() *rest.Config {
	log.SetLogger(zap.New(zap.WriteTo(ginkgo.GinkgoWriter), zap.UseDevMode(true)))
	ginkgo.By("bootstrapping test environment")
	f.testEnv = &envtest.Environment{
		CRDDirectoryPaths: []string{filepath.Join("..", "..", "..", "manifests", "v2", "base", "crds")},
		WebhookInstallOptions: envtest.WebhookInstallOptions{
			Paths: []string{filepath.Join("..", "..", "..", "manifests", "v2", "base", "webhook")},
		},
		ErrorIfCRDPathMissing: true,
	}
	cfg, err := f.testEnv.Start()
	gomega.Expect(err).NotTo(gomega.HaveOccurred())
	gomega.Expect(cfg).NotTo(gomega.BeNil())
	return cfg
}

func (f *Framework) RunManager(cfg *rest.Config) (context.Context, client.Client) {
	webhookInstallOpts := &f.testEnv.WebhookInstallOptions
	gomega.ExpectWithOffset(1, kubeflowv2.AddToScheme(scheme.Scheme)).NotTo(gomega.HaveOccurred())

	// +kubebuilder:scaffold:scheme

	k8sClient, err := client.New(cfg, client.Options{Scheme: scheme.Scheme})
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred())
	gomega.ExpectWithOffset(1, k8sClient).NotTo(gomega.BeNil())

	ctx, cancel := context.WithCancel(context.Background())
	f.cancel = cancel
	mgr, err := ctrl.NewManager(cfg, manager.Options{
		Scheme: scheme.Scheme,
		Metrics: metricsserver.Options{
			BindAddress: "0", // disable metrics to avoid conflicts between packages.
		},
		WebhookServer: webhook.NewServer(
			webhook.Options{
				Host:    webhookInstallOpts.LocalServingHost,
				Port:    webhookInstallOpts.LocalServingPort,
				CertDir: webhookInstallOpts.LocalServingCertDir,
			}),
	})
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred(), "failed to create manager")

	failedCtrlName, err := controllerv2.SetupControllers(mgr)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred(), "controller", failedCtrlName)
	failedWebhookName, err := webhookv2.Setup(mgr)
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred(), "webhook", failedWebhookName)

	go func() {
		defer ginkgo.GinkgoRecover()
		err = mgr.Start(ctx)
		gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred(), "failed to run manager")
	}()

	dialer := &net.Dialer{Timeout: time.Second}
	addrPort := fmt.Sprintf("%s:%d", webhookInstallOpts.LocalServingHost, webhookInstallOpts.LocalServingPort)
	gomega.Eventually(func(g gomega.Gomega) {
		var conn *tls.Conn
		conn, err = tls.DialWithDialer(dialer, "tcp", addrPort, &tls.Config{InsecureSkipVerify: true})
		g.Expect(err).Should(gomega.Succeed())
		g.Expect(conn.Close()).Should(gomega.Succeed())
	}).Should(gomega.Succeed())
	return ctx, k8sClient
}

func (f *Framework) Teardown() {
	ginkgo.By("tearing down the test environment")
	if f.cancel != nil {
		f.cancel()
	}
	gomega.ExpectWithOffset(1, f.testEnv.Stop()).NotTo(gomega.HaveOccurred())
}
