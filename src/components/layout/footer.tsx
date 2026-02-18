import Link from "next/link";
import { GraduationCap, Linkedin } from "lucide-react";

const footerLinks = {
  product: [
    { href: "/roadmaps", label: "Roadmaps" },
    { href: "/courses", label: "Courses" },
    { href: "/guides", label: "Guides" },
    { href: "/blog", label: "Blog" },
    { href: "/projects", label: "Projects" },
    { href: "/tutor", label: "AI Tutor" },
    { href: "/search", label: "Search" },
  ],
  company: [
    { href: "/about", label: "About" },
    { href: "/pricing", label: "Pricing" },
    { href: "/faq", label: "FAQ" },
    { href: "/contact", label: "Contact" },
  ],
  auth: [
    { href: "/auth/signin", label: "Sign in" },
    { href: "/auth/signup", label: "Sign up" },
  ],
};

export function Footer() {
  return (
    <footer className="border-t bg-muted/20 mt-auto">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-10 sm:py-12 max-w-7xl 2xl:max-w-8xl">
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-6 sm:gap-8">
          <div className="col-span-2 sm:col-span-1">
            <Link href="/" className="flex items-center gap-2 font-bold text-lg">
              <GraduationCap className="h-6 w-6 text-primary" />
              AICourseMap
            </Link>
            <p className="mt-2 text-sm text-muted-foreground">
              Grow your AI career with structured learning paths.
            </p>
          </div>
          <div>
            <h4 className="font-semibold mb-3">Product</h4>
            <ul className="space-y-2">
              {footerLinks.product.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-muted-foreground hover:text-foreground"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h4 className="font-semibold mb-3">Company</h4>
            <ul className="space-y-2">
              {footerLinks.company.map((link) => (
                <li key={link.href}>
                  <Link
                    href={link.href}
                    className="text-sm text-muted-foreground hover:text-foreground"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>
        <div className="mt-8 pt-8 border-t flex flex-col sm:flex-row items-center justify-center gap-2 sm:gap-4 text-sm text-muted-foreground">
          <span>© {new Date().getFullYear()} AICourseMap. All rights reserved.</span>
          <a
            href="https://www.linkedin.com/in/hari-mohan-47299b54/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 hover:text-foreground transition-colors"
          >
            <Linkedin className="h-4 w-4" />
            Connect on LinkedIn
          </a>
        </div>
      </div>
    </footer>
  );
}
